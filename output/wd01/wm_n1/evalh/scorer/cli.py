"""The CAL-tune scorer CLI (E4): ``python -m evalh.scorer.cli``.

Subcommands:

* ``score`` -- one pass over the released CAL-tune half: bind the pinned
  member-1 checkpoint, run the A / v2g predictors, write ``RESULTS.json``,
  a per-row CSV and a ``DONE.json`` receipt carrying every input SHA-256
  (release specification, checkpoint bytes, comparator params, and the code
  commit via ``git rev-parse HEAD``).
* ``dry-run`` -- validate release/checkpoint metadata only and print the
  planned reads; no data shard, no payload, no output directory.

Fail-closed guards: the output directory must not exist (one pass), the
released half must be ``cal_tune``, and the member file must match the
release spec's ``checkpoint_member_sha256``.  Predictor construction is
injectable: the real factories come from ``evalh.scorer.actions`` (model A)
and ``evalh.scorer.comparator_v2g`` (Part A v2g), both built in parallel and
imported lazily here; tests inject fakes.  No CAL data is opened by this
module's own tests (generated fixtures only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evalh.cal_split import CAL_TUNE
from evalh.scorer.checkpoint import (EXPECTED_MEMBER_SHA256,
                                     DYNAMICS_ROLLOUT_LABEL)

SCHEMA_RESULTS = "wm_n1_scorer_results/1"
SCHEMA_DONE = "wm_n1_scorer_done/1"
SCHEMA_DRY_RUN = "wm_n1_scorer_dry_run/1"

_TAUS = (30, 60, 120, 300, 570)


class ScorerCliRefusal(RuntimeError):
    """The scoring run was refused before or during execution."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _required(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScorerCliRefusal(f"{label} must be explicit")
    return value.strip()


# ------------------------------------------------------------ lazy factories --
def _default_source_factory(release_spec, *, expected_release_sha256):
    from evalh.real_source import ReleasedCalSource

    return ReleasedCalSource(release_spec,
                             expected_release_sha256=expected_release_sha256)


def _default_checkpoint_loader(member_path, *, expected_sha256=None):
    from evalh.scorer.checkpoint import load_scoring_checkpoint

    if expected_sha256 is None:
        return load_scoring_checkpoint(member_path)
    return load_scoring_checkpoint(member_path,
                                   expected_sha256=expected_sha256)


def _default_rows_loader(source):
    try:
        from evalh.scorer.labels import tune_outcome_rows
    except ImportError as exc:                           # pragma: no cover
        raise ScorerCliRefusal(
            "evalh.scorer.labels (E3) is not available yet; inject "
            "rows_loader") from exc
    return tune_outcome_rows(source)


def _cav_motions() -> tuple:
    from evalh.scorer.predictors import CAV_MOTIONS

    return CAV_MOTIONS


def _default_a_factory(bundle, *, source, cav_motion,
                       dynamics_extrapolation=False):
    """The real per-item model A predictor (E7), explicit CAV-motion mode."""
    from evalh.scorer.predictors import ModelAPredictor

    if cav_motion not in _cav_motions():
        raise ScorerCliRefusal(
            f"--cav-motion must be one of {list(_cav_motions())}; got "
            f"{cav_motion!r}")
    return ModelAPredictor(bundle, source=source, cav_motion=cav_motion,
                           dynamics_extrapolation=dynamics_extrapolation)


def _default_v2g_factory(source, *, bundle, option, cav_motion,
                         engine_sha256=None):
    """The real Part A v2g predictor (E7), explicit owner delay option."""
    from evalh.scorer.comparator_v2g import bind_v2g_artifacts
    from evalh.scorer.predictors import build_v2g_predictor

    if cav_motion not in _cav_motions():
        raise ScorerCliRefusal(
            f"--cav-motion must be one of {list(_cav_motions())}; got "
            f"{cav_motion!r}")
    if not option:
        raise ScorerCliRefusal(
            "--v2g-delay-option is required (no default)")
    root = Path(__file__).resolve().parents[3] / "partA_v2g"
    artifacts = bind_v2g_artifacts(root, expected_engine_sha256=engine_sha256)
    return build_v2g_predictor(source, bundle, option=option,
                               cav_motion=cav_motion, artifacts=artifacts)


def _default_git_head() -> str:
    root = Path(__file__).resolve().parents[4]           # .../scorer-e4
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=True)
    except Exception as exc:                             # noqa: BLE001
        raise ScorerCliRefusal(
            "cannot resolve the code commit via git rev-parse HEAD; pass "
            "git_head explicitly") from exc
    return out.stdout.strip()


# ------------------------------------------------------------------ helpers --
def _bind_source(args, source_factory):
    sha = _required(args.release_sha, "raw release SHA-256")
    if (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
        raise ScorerCliRefusal(
            "release sha must be 64 lowercase hex digits")
    spec_path = _required(args.release_spec, "release specification path")
    if source_factory is None:
        source_factory = _default_source_factory
    return source_factory(spec_path, expected_release_sha256=sha)


def _require_cal_tune(source) -> None:
    half = getattr(source, "half", None)
    if half != CAL_TUNE:
        raise ScorerCliRefusal(
            f"scorer runs only on the {CAL_TUNE!r} half; released half is "
            f"{half!r}")


def _census_dict(census) -> dict:
    if census is None:
        return {}
    if isinstance(census, dict):
        return dict(census)
    import dataclasses

    if dataclasses.is_dataclass(census):
        return dataclasses.asdict(census)
    raise ScorerCliRefusal(
        f"rows loader census must be a mapping or dataclass; got "
        f"{type(census).__name__}")


def _excluded_b_count(census) -> int:
    """Owner rule-(b) exclusion count from a rows-loader census (0 if absent)."""
    if census is None:
        return 0
    if isinstance(census, dict):
        return int(census.get("n_excluded_b", 0) or 0)
    return int(getattr(census, "n_excluded_b", 0) or 0)


def _params_sha(args, v2g_predictor) -> str:
    sha = getattr(args, "params_sha256", None)
    if sha:
        sha = str(sha).strip()
    else:
        sha = str(getattr(v2g_predictor, "params_lf_sha256", "") or "").strip()
    if (len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)):
        raise ScorerCliRefusal(
            "the comparator params SHA-256 is unavailable: pass "
            "--params-sha256 or expose params_lf_sha256 on the v2g "
            "predictor; the DONE receipt refuses incomplete input hashes")
    return sha


def _engine_sha(v2g_predictor):
    """The frozen v2g engine SHA-256, recorded when the predictor exposes it."""
    artifacts = getattr(v2g_predictor, "artifacts", None)
    if isinstance(artifacts, dict):
        sha = str(artifacts.get("engine_sha256", "") or "").strip()
        if sha:
            return sha
    return None


def _mode_inputs(args) -> dict:
    """The explicit owner decisions this run recorded (fail-closed modes)."""
    extrapolation = bool(getattr(args, "dynamics_extrapolation", False))
    return dict(cav_motion=getattr(args, "cav_motion", None),
                v2g_delay_option=getattr(args, "v2g_delay_option", None),
                dynamics_extrapolation=extrapolation,
                v2g_engine_sha256=getattr(args, "v2g_engine_sha256", None))


#: The registered comparator set for this run (owner option (c)).  Rule-based
#: v2g is deliberately NOT a member: it is registered later and refused here.
_COMPARATORS = ("zero", "train_climatology")


def _parse_comparators(raw) -> list:
    """Parse ``--comparators``; refuse empty, unknown, or rule-based 'v2g'."""
    value = _required(raw, "--comparators")
    names = [n.strip().lower() for n in value.split(",") if n.strip()]
    if not names:
        raise ScorerCliRefusal("--comparators must name at least one comparator")
    if "v2g" in names:
        raise ScorerCliRefusal(
            "'v2g' is refused in --comparators; rule-based v2g is registered "
            "later, not in this run")
    unknown = [n for n in names if n not in _COMPARATORS]
    if unknown:
        raise ScorerCliRefusal(
            f"unknown comparator(s) {sorted(set(unknown))}; the registered set "
            f"is {list(_COMPARATORS)}")
    seen: list = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def _climatology_inputs(args) -> tuple:
    """The explicit climatology path + SHA-256 pin (both required)."""
    path = _required(getattr(args, "climatology", None), "--climatology")
    sha = _required(getattr(args, "climatology_sha256", None),
                    "--climatology-sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ScorerCliRefusal(
            "--climatology-sha256 must be 64 lowercase hex digits")
    return path, sha


def _build_comparators(names, args) -> dict:
    """Build the direct-dV comparator models for one run (fail closed)."""
    from evalh.scorer.baselines import make_comparator

    clim_path = clim_sha = None
    if "train_climatology" in names:
        clim_path, clim_sha = _climatology_inputs(args)
    models = {}
    for name in names:
        model = make_comparator(name, climatology=clim_path,
                                climatology_sha256=clim_sha)
        models[name] = model
    return models, clim_path, clim_sha


def _decision_quality_inputs(args) -> tuple:
    """The explicit fixed-best path + SHA-256 pin (both required)."""
    path = _required(getattr(args, "dq_fixed_best", None), "--dq-fixed-best")
    sha = _required(getattr(args, "dq_fixed_best_sha256", None),
                    "--dq-fixed-best-sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ScorerCliRefusal(
            "--dq-fixed-best-sha256 must be 64 lowercase hex digits")
    return path, sha


#: Decision-quality arm-prediction sources (``--dq-source``).
DQ_SOURCES = ("marginals", "paired_head")

#: Decision-quality candidate arm registries (``--dq-arms``).
DQ_ARM_SETS = ("registered", "e2")


def _dq_allowed_arms(args) -> tuple:
    """The registered DQ candidate arms (``--dq-arms``; default unchanged)."""
    from evalh.scorer import arm_contrasts as AC
    from evalh.scorer.decision_quality import ALLOWED_ARMS

    choice = str(getattr(args, "dq_arms", None) or "registered").strip().lower()
    if choice not in DQ_ARM_SETS:
        raise ScorerCliRefusal(
            f"unknown --dq-arms {choice!r}; registered choices are "
            f"{list(DQ_ARM_SETS)}")
    return tuple(ALLOWED_ARMS) if choice == "registered" else tuple(
        AC.E2_DQ_CANDIDATE_ARMS)


def _dq_normalizer(args):
    """The ``--dq-arms e2`` row->candidate normalizer (default ``None``).

    Fix 3 (2026-10-05): each trajectory arm (``arm0``..``arm3``) stays its own
    candidate rather than being pooled into a ``trajectory`` candidate.
    """
    choice = str(getattr(args, "dq_arms", None) or "registered").strip().lower()
    if choice != "e2":
        return None
    from evalh.scorer.decision_quality import _e2_candidate_arm

    return _e2_candidate_arm


def _dq_pool_arms(args):
    """``--dq-arms e2`` labels pooled by mean (random mixes); else ``None``."""
    choice = str(getattr(args, "dq_arms", None) or "registered").strip().lower()
    if choice != "e2":
        return None
    from evalh.scorer.decision_quality import E2_POOL_ARMS

    return tuple(sorted(E2_POOL_ARMS))


def _dq_dedup_key(args):
    """``--dq-arms e2`` duplicate preference (truearms over released)."""
    choice = str(getattr(args, "dq_arms", None) or "registered").strip().lower()
    if choice != "e2":
        return None
    from evalh.scorer.decision_quality import e2_dedup_key

    return e2_dedup_key


def _extra_cal_rows(args, source, taus, extra_pack=None) -> tuple:
    """Validate and materialise the sealed ``--extra-cal-labels`` packs.

    Each pack's ``(world, arm)`` must name a registered E2 arm on a released
    CAL-tune world; anything else (unknown arm, foreign world, missing
    ``SHA256SUMS``, unmanifested file) is refused.  Returns
    ``(rows, receipt, keys)`` or ``(None, None, None)`` when no pack was
    supplied.  ``rows`` are :class:`evalh.protocol.OutcomeRow` objects whose
    ``pack_item_position`` resolves either to the sealed extra feature pack
    (trajectory arms) or to the released CAL shard (fixed/switching/wall arms);
    ``keys`` are the ``(world, t0, dec, tau, arm_type)`` rows the sealed labels
    override in the released rows.
    """
    paths = getattr(args, "extra_cal_labels", None)
    if not paths:
        return None, None, None
    from evalh.protocol import OutcomeRow
    from evalh.scorer import arm_contrasts as AC
    from evalh.scorer.extra_labels import load_extra_cal_labels

    base = getattr(source, "base", source)
    worlds = {str(item["world"]) for _pos, item, _run
              in (getattr(base, "selected", ()) or ())}
    if not worlds:
        raise ScorerCliRefusal(
            "--extra-cal-labels requires a released CAL-tune selected list")
    selected = {(w, arm) for w in worlds for arm in AC.E2_ALLOWED_ARMS}
    records, receipt = load_extra_cal_labels(
        paths, selected_keys=selected, taus=taus,
        allowed_sources=AC.E2_ALLOWED_ARMS)

    released_pos = {}
    for pos, item, _run in (getattr(base, "selected", ()) or ()):
        released_pos[(str(item["world"]), str(item["arm"]))] = int(pos)

    rows, keys = [], set()
    for rec in records:
        world = str(rec.get("world", ""))
        arm = str(rec.get("arm", "") or rec.get("source", ""))
        position = None
        if extra_pack is not None:
            position = extra_pack.positions.get((world, arm))
        if position is None:
            position = released_pos.get((world, arm))
        if position is None:
            raise ScorerCliRefusal(
                f"sealed extra label {world}/{arm} has no feature source "
                "(neither the released CAL shards nor --extra-cal-pack)")
        policy = AC.arm_type_of(arm)
        t0 = round(float(rec["t0_s"]), 6)
        dec = round(float(rec.get("t", t0) or t0), 6)
        tau = round(float(rec["tau"]), 6)
        rows.append(OutcomeRow(
            world=world, t0_s=t0, decision_t_s=dec, tau_s=tau, policy=policy,
            y_delay_s=float(rec["total_delay_ref_s"]),
            collision=float(rec.get("collision_any", 0.0) or 0.0),
            ttc_lt3_s=float(rec.get("ttc_lt3_veh_s", 0.0) or 0.0),
            hard_brake_s=float(rec.get("hard_brake_s", 0.0) or 0.0),
            n_vehicles=float(rec.get("n_vehicles_present", 0.0) or 0.0),
            features={"pack_item_position": int(position), "arm": arm,
                      "run_key": f"{world}__{arm}"}))
        keys.add((world, t0, dec, tau, policy))
    if not rows:
        raise ScorerCliRefusal(
            "no usable sealed extra CAL-tune label rows were produced")
    return rows, receipt, keys


def _load_extra_pack_inputs(args, source):
    """The optional sealed ``--extra-cal-pack`` + SHA pin (fail closed)."""
    path = getattr(args, "extra_cal_pack", None)
    if not path:
        return None
    from evalh.scorer import arm_contrasts as AC
    from evalh.scorer.extra_pack import ExtraPackSource, load_extra_pack

    sha = _required(getattr(args, "extra_cal_pack_sha256", None),
                    "--extra-cal-pack-sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ScorerCliRefusal(
            "--extra-cal-pack-sha256 must be 64 lowercase hex digits")
    worlds = {str(item["world"]) for _pos, item, _run
              in (getattr(source, "selected", ()) or ())}
    if not worlds:
        raise ScorerCliRefusal(
            "--extra-cal-pack requires a released CAL-tune selected list")
    arms = tuple(a for a in AC.ALL_ARMS if a.startswith("arm"))
    extra = load_extra_pack(path, expected_sha256=sha, worlds=worlds, arms=arms)
    return ExtraPackSource(source, extra)


def _decision_quality_predictions(rows, q_a, args, predictor_a):
    """The per-arm prediction matrix for ``--dq-source`` (fail closed)."""
    source = str(getattr(args, "dq_source", None) or "marginals").strip().lower()
    if source not in DQ_SOURCES:
        raise ScorerCliRefusal(
            f"unknown --dq-source {source!r}; registered sources are "
            f"{list(DQ_SOURCES)}")
    if source == "marginals":
        return q_a, source
    if predictor_a is None or getattr(predictor_a, "paired_head_dv", None) is None:
        raise ScorerCliRefusal(
            "--dq-source paired_head requires a model A predictor exposing "
            "'paired_head_dv' (the optional climatology-anchored paired-effect "
            "head); this checkpoint/release exposes none")
    import numpy as np

    q_dv = np.asarray(predictor_a.predict_paired_head_row_quantiles(rows),
                      dtype=float)
    return q_dv, source


def _run_decision_quality(rows, q_a, args, predictor_a=None) -> tuple:
    """Build the decision-quality doc + markdown table (fail closed)."""
    from evalh.scorer.decision_quality import (COSTS, TAUS_S,
                                               decision_points_from_rows,
                                               evaluate, load_fixed_best,
                                               render_table)

    path, sha = _decision_quality_inputs(args)
    fixed_doc, sha = load_fixed_best(path, expected_sha256=sha)
    q_used, source = _decision_quality_predictions(rows, q_a, args, predictor_a)
    allowed_arms = _dq_allowed_arms(args)
    normalizer = _dq_normalizer(args)
    pool_arms = _dq_pool_arms(args)
    dedup_key = _dq_dedup_key(args)
    points = []
    for cost in COSTS:
        cost_points, _stats = decision_points_from_rows(
            rows, q_used, cost=cost, taus=TAUS_S, allowed_arms=allowed_arms,
            normalizer=normalizer, pool_arms=pool_arms, dedup_key=dedup_key)
        points.extend(cost_points)
    if not points:
        raise ScorerCliRefusal(
            "no decision points survive the fail-closed filters; refusing an "
            "empty decision-quality report")
    n_boot = int(getattr(args, "dq_n_boot", None) or 2000)
    doc = evaluate(points, fixed_best=fixed_doc, n_boot=n_boot, source=source)
    inputs = dict(fixed_best=path, fixed_best_sha256=sha, n_boot=n_boot,
                  dq_source=source, dq_arms=list(allowed_arms))
    return doc, render_table(doc), inputs


def _write_rows_csv(path: Path, rows, q_by_model: dict, levels) -> None:
    names = list(q_by_model)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        head = ["world", "t0_s", "decision_t_s", "tau_s", "policy",
                "y_delay_s", "collision", "veh_steps", "ttc_lt3_s",
                "hard_brake_s", "n_vehicles", "n_arrivals", "intervention"]
        for name in names:
            head += [f"q_{name}_q{int(round(lv * 100)):02d}" for lv in levels]
        writer.writerow(head)
        for i, r in enumerate(rows):
            writer.writerow([
                r.world, repr(float(r.t0_s)), repr(float(r.decision_t_s)),
                repr(float(r.tau_s)), r.policy, repr(float(r.y_delay_s)),
                repr(float(r.collision)), repr(float(r.veh_steps)),
                repr(float(r.ttc_lt3_s)), repr(float(r.hard_brake_s)),
                repr(float(r.n_vehicles)), repr(float(r.n_arrivals)),
                repr(float(r.intervention))]
                + [repr(float(q_by_model[name][i][j]))
                   for name in names for j in range(len(levels))])


# ------------------------------------------------------------------- score --
def run_score(args, *, source_factory=None, checkpoint_loader=None,
              a_factory=None, v2g_factory=None, rows_loader=None,
              git_head=None) -> dict:
    """One scoring pass; every real reader is injectable for tests."""
    out_dir = Path(_required(args.out, "output directory"))
    if out_dir.exists():
        raise ScorerCliRefusal(
            f"output directory {out_dir} already exists; the scorer runs "
            "one pass and never overwrites")
    source = _bind_source(args, source_factory)
    _require_cal_tune(source)
    source = _load_extra_pack_inputs(args, source) or source
    extra_pack = getattr(source, "extra", None)
    # ``--comparators`` is REQUIRED (no default); ``main`` refuses a missing
    # list before this point.  The historical A-vs-v2g path is retained for the
    # injected tests that predate option (c); the option-(c) path never requires
    # ``--v2g-delay-option``.
    cav_motion = getattr(args, "cav_motion", None)
    comparators_raw = getattr(args, "comparators", None)
    use_comparators = comparators_raw is not None
    comp_names: list = []
    if use_comparators:
        comp_names = _parse_comparators(comparators_raw)
        if a_factory is None and cav_motion not in _cav_motions():
            raise ScorerCliRefusal(
                f"--cav-motion is required and must be one of "
                f"{list(_cav_motions())}; got {cav_motion!r}")
    else:
        v2g_option = getattr(args, "v2g_delay_option", None)
        if a_factory is None and cav_motion not in _cav_motions():
            raise ScorerCliRefusal(
                f"--cav-motion is required and must be one of "
                f"{list(_cav_motions())}; got {cav_motion!r}")
        if v2g_factory is None:
            if cav_motion not in _cav_motions():
                raise ScorerCliRefusal(
                    f"--cav-motion is required and must be one of "
                    f"{list(_cav_motions())}; got {cav_motion!r}")
            if not v2g_option:
                raise ScorerCliRefusal(
                    "--v2g-delay-option is required (no default owner "
                    "option)")
    member_path = Path(_required(args.checkpoint, "checkpoint member path"))
    if not member_path.is_file():
        raise ScorerCliRefusal(f"checkpoint member missing: {member_path}")
    if checkpoint_loader is None:
        checkpoint_loader = _default_checkpoint_loader
    spec = getattr(source, "spec", None) or {}
    spec_sha = str(spec.get("checkpoint_member_sha256", "") or "")
    # The frozen release spec's checkpoint pin is the authority (A8 scoring,
    # 2026-10-04); the module default pin only applies when the spec has none.
    if spec_sha and checkpoint_loader is _default_checkpoint_loader:
        bound = checkpoint_loader(member_path, expected_sha256=spec_sha)
    else:
        bound = checkpoint_loader(member_path)
    actual_sha = getattr(bound, "sha256", None) or _sha256_file(member_path)
    if spec_sha and spec_sha != actual_sha:
        raise ScorerCliRefusal(
            f"checkpoint member sha256 {actual_sha} != release-spec pin "
            f"{spec_sha}")
    if rows_loader is None:
        rows_loader = _default_rows_loader
    # The released shard reader must never see an extra-pack item.
    rows, census = rows_loader(getattr(source, "base", source))
    _extra_rows, extra_receipt, extra_keys = _extra_cal_rows(
        args, source, _TAUS, extra_pack)
    if _extra_rows:
        from evalh.scorer.decision_quality import _row_arm_type

        rows = [r for r in rows
                if (str(r.world), round(float(r.t0_s), 6),
                    round(float(r.decision_t_s), 6), round(float(r.tau_s), 6),
                    _row_arm_type(r)) not in extra_keys]
        rows = list(rows) + list(_extra_rows)
    bundle = getattr(bound, "bundle", bound)
    if a_factory is None:
        extrapolation = bool(getattr(args, "dynamics_extrapolation", False))
        a_factory = lambda b: _default_a_factory(          # noqa: E731
            b, source=source, cav_motion=cav_motion,
            dynamics_extrapolation=extrapolation)
    predictor_a = a_factory(bundle)
    from evalh.scorer.metrics_loop import (DEFAULT_DV_SAMPLES,
                                           DV_METHOD_COMONOTONE,
                                           DV_METHOD_PAIRED_HEAD,
                                           score as _score)

    import numpy as np

    dv_method = str(getattr(args, "dv_method", None) or DV_METHOD_COMONOTONE)
    dv_samples = int(getattr(args, "dv_samples", None) or DEFAULT_DV_SAMPLES)
    q_a = np.asarray(predictor_a.predict_quantiles(rows), dtype=float)
    code_commit = git_head if git_head is not None else _default_git_head()
    release_sha = _required(args.release_sha, "raw release SHA-256")
    if use_comparators:
        comparators, clim_path, clim_sha = _build_comparators(comp_names, args)
        paired_head = None
        if dv_method == DV_METHOD_PAIRED_HEAD:
            paired_head = getattr(predictor_a, "paired_head_dv", None)
            if paired_head is None:
                raise ScorerCliRefusal(
                    "--dv-method paired_head requires a model A predictor "
                    "exposing 'paired_head_dv' (the optional "
                    "climatology-anchored paired-effect head); this "
                    "checkpoint/release exposes none")
        metrics = _score(rows, {"A": q_a}, dv_models=comparators, taus=_TAUS,
                         dv_method=dv_method, dv_samples=dv_samples,
                         paired_head=paired_head)
        q_by_model = {"A": q_a}
        predictors = {"A": str(getattr(predictor_a, "name", "A"))}
        results_extra = dict(comparators=list(comp_names))
        inputs = dict(
            release_spec=str(args.release_spec),
            release_spec_sha256=release_sha,
            checkpoint=str(member_path),
            checkpoint_sha256=actual_sha,
            code_commit=code_commit,
            comparators=list(comp_names),
            climatology=str(clim_path) if clim_path else None,
            climatology_sha256=clim_sha,
            cav_motion=cav_motion,
            dv_method=dv_method,
            dv_samples=dv_samples,
            paired_head_source=(None if paired_head is None else "model_a"),
            summary_source=getattr(predictor_a, "summary_source", None))
    else:
        v2g_option = getattr(args, "v2g_delay_option", None)
        if v2g_factory is None:
            v2g_factory = lambda s: _default_v2g_factory(  # noqa: E731
                s, bundle=bundle, option=v2g_option, cav_motion=cav_motion,
                engine_sha256=getattr(args, "v2g_engine_sha256", None))
        predictor_v2g = v2g_factory(source)
        params_sha = _params_sha(args, predictor_v2g)
        q_v2g = np.asarray(predictor_v2g.predict_quantiles(rows), dtype=float)
        metrics = _score(rows, {"A": q_a, "v2g": q_v2g}, taus=_TAUS,
                         dv_method=dv_method, dv_samples=dv_samples)
        q_by_model = {"A": q_a, "v2g": q_v2g}
        predictors = {"A": str(getattr(predictor_a, "name", "A")),
                      "v2g": str(getattr(predictor_v2g, "name", "v2g"))}
        results_extra = {}
        inputs = dict(
            release_spec=str(args.release_spec),
            release_spec_sha256=release_sha,
            checkpoint=str(member_path),
            checkpoint_sha256=actual_sha,
            params_sha256=params_sha,
            engine_sha256=_engine_sha(predictor_v2g),
            code_commit=code_commit,
            **_mode_inputs(args))
        inputs["dv_method"] = dv_method
        inputs["dv_samples"] = dv_samples
    if extra_receipt is not None:
        inputs["extra_cal_labels"] = extra_receipt
    if extra_pack is not None:
        inputs["extra_cal_pack"] = dict(
            path=str(extra_pack.root),
            manifest_sha256=extra_pack.index.get("manifest_sha256"),
            index_sha256=extra_pack.index.get("sha256sums_sha256"),
            n_items=len(extra_pack.selected),
            worlds=sorted(extra_pack.worlds))
    dq_doc = dq_table = None
    if bool(getattr(args, "decision_quality", False)):
        dq_doc, dq_table, dq_inputs = _run_decision_quality(
            rows, q_a, args, predictor_a=predictor_a)
        inputs["decision_quality"] = dq_inputs
    out_dir.mkdir(parents=True)
    if dq_doc is not None:
        (out_dir / "DECISION_QUALITY.json").write_text(
            json.dumps(dq_doc, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")

    results = dict(
        schema=SCHEMA_RESULTS,
        written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cal_half=getattr(source, "half", None),
        inputs=inputs,
        predictors=predictors,
        census=_census_dict(census),
        dynamics_rollout_label=DYNAMICS_ROLLOUT_LABEL,
        metrics=metrics,
        **results_extra)
    if dq_doc is not None:
        results["decision_quality"] = dq_doc
        results["decision_quality_table"] = dq_table
    extrapolation = getattr(predictor_a, "extrapolation_report", None)
    if extrapolation is not None:
        results["dynamics_extrapolation"] = extrapolation
    (out_dir / "RESULTS.json").write_text(
        json.dumps(results, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    from evalh.protocol import QUANTILE_LEVELS

    _write_rows_csv(out_dir / "rows.csv", rows, q_by_model, QUANTILE_LEVELS)
    done = dict(
        schema=SCHEMA_DONE,
        written_utc=results["written_utc"],
        cal_half=getattr(source, "half", None),
        out_dir=str(out_dir),
        n_rows=len(rows),
        n_pairs=metrics["counts"]["n_pairs"],
        n_excluded_b=_excluded_b_count(census),
        census=_census_dict(census),
        inputs=inputs)
    (out_dir / "DONE.json").write_text(
        json.dumps(done, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return done


# ----------------------------------------------------------------- dry run --
def run_dry_run(args, *, source_factory=None) -> dict:
    """Validate release/checkpoint metadata only; print planned reads."""
    source = _bind_source(args, source_factory)
    _require_cal_tune(source)
    member_path = Path(_required(args.checkpoint, "checkpoint member path"))
    if not member_path.is_file():
        raise ScorerCliRefusal(f"checkpoint member missing: {member_path}")
    actual = _sha256_file(member_path)
    spec = getattr(source, "spec", None) or {}
    spec_sha = str(spec.get("checkpoint_member_sha256", "") or "")
    if spec_sha:
        if spec_sha != actual:
            raise ScorerCliRefusal(
                f"checkpoint member sha256 {actual} != release-spec pin "
                f"{spec_sha}")
    elif actual != EXPECTED_MEMBER_SHA256:
        raise ScorerCliRefusal(
            f"checkpoint member sha256 {actual} != pinned "
            f"{EXPECTED_MEMBER_SHA256} and the release spec records no "
            "checkpoint pin")
    selected = getattr(source, "selected", None)
    n_runs = len(selected) if selected is not None else None
    return dict(
        schema=SCHEMA_DRY_RUN,
        status="metadata_valid_no_payload_read",
        cal_half=getattr(source, "half", None),
        planned_reads=dict(
            release_spec=str(args.release_spec),
            release_spec_sha256=_required(args.release_sha,
                                          "raw release SHA-256"),
            checkpoint=str(member_path),
            checkpoint_sha256=actual,
            selected_run_count=n_runs,
            outcomes_payload_read=False,
            torch_deserialization_performed=False))


# -------------------------------------------------------------------- main --
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evalh.scorer.cli",
        description="WM-N1 CAL-tune scorer: model A vs Part A v2g")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
            ("score", "one scoring pass over the released CAL-tune half"),
            ("dry-run", "validate metadata only; no payload read")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--release-spec", required=True,
                       help="externally registered release JSON path")
        p.add_argument("--release-sha", required=True,
                       help="externally registered raw SHA-256 of the spec")
        p.add_argument("--checkpoint", required=True,
                       help="member checkpoint path (member_1.pt)")
        if name == "score":
            from evalh.scorer.comparator_v2g import _DELAY_OPTIONS

            p.add_argument("--out", required=True,
                           help="output directory (must not exist)")
            p.add_argument("--comparators", default=None,
                           help="REQUIRED comma list of registered comparators "
                                "from {zero, train_climatology} (no default); "
                                "'v2g' is refused (registered later)")
            p.add_argument("--climatology", default=None,
                           help="TRAIN climatology JSON path (required when "
                                "'train_climatology' is listed)")
            p.add_argument("--climatology-sha256", dest="climatology_sha256",
                           default=None,
                           help="SHA-256 pin of the climatology JSON (required "
                                "when 'train_climatology' is listed)")
            p.add_argument("--params-sha256", default=None,
                           help="(historical A-vs-v2g path) comparator params "
                                "LF-sha256 for the DONE receipt")
            p.add_argument("--cav-motion", dest="cav_motion", default=None,
                           choices=list(_cav_motions()),
                           help="REQUIRED explicit CAV-motion mode: "
                                "'controller' (packed per-CAV schedule) or "
                                "'teacher' (recorded CAV kinematics); no "
                                "default, refused when omitted")
            p.add_argument("--v2g-delay-option", dest="v2g_delay_option",
                           default=None, choices=list(_DELAY_OPTIONS),
                           help="(historical A-vs-v2g path) explicit Part A "
                                "v2g delay option; not required when "
                                "--comparators is given")
            p.add_argument("--dynamics-extrapolation",
                           dest="dynamics_extrapolation",
                           action="store_true",
                           help="also run a longer dynamics rollout, reported "
                                 "separately as 'extrapolation'; never feeds "
                                 "the head")
            p.add_argument("--dv-method", dest="dv_method",
                           default="comonotone",
                           choices=["comonotone", "independent",
                                    "paired_head"],
                           help="paired-dV construction for per-arm marginal "
                                "predictors: 'comonotone' (default, historical "
                                "same-level q_a-q_b), 'independent' (sample "
                                "each marginal, take the difference "
                                "quantiles; honest when arms are independent) "
                                "or 'paired_head' (model A dV comes directly "
                                "from the optional climatology-anchored "
                                "paired-effect head; requires a predictor "
                                "exposing paired_head_dv)")
            p.add_argument("--dv-samples", dest="dv_samples", type=int,
                           default=4096,
                           help="samples per marginal for --dv-method "
                                "independent (default 4096)")
            p.add_argument("--v2g-engine-sha256", dest="v2g_engine_sha256",
                           default=None,
                           help="optional pin of the frozen v2g engine bytes")
            p.add_argument("--decision-quality", dest="decision_quality",
                           action="store_true",
                           help="also score decision-quality (regret) and write "
                                 "DECISION_QUALITY.json + a table in the score "
                                 "report; default off")
            p.add_argument("--dq-fixed-best", dest="dq_fixed_best", default=None,
                           help="TRAIN-only frozen fixed-best JSON path "
                                "(required with --decision-quality)")
            p.add_argument("--dq-fixed-best-sha256",
                           dest="dq_fixed_best_sha256", default=None,
                           help="SHA-256 pin of the fixed-best JSON (required "
                                "with --decision-quality)")
            p.add_argument("--dq-n-boot", dest="dq_n_boot", type=int,
                           default=2000,
                           help="world-bootstrap draws for decision-quality "
                                "CIs (default 2000)")
            p.add_argument("--dq-source", dest="dq_source",
                           default="marginals", choices=list(DQ_SOURCES),
                           help="decision-quality arm predictions: 'marginals' "
                                "(default, per-arm marginal quantiles) or "
                                "'paired_head' (per-arm predicted dV vs native "
                                "from the optional paired-effect head; native "
                                "== 0); default unchanged")
            p.add_argument("--dq-arms", dest="dq_arms",
                           default="registered", choices=list(DQ_ARM_SETS),
                           help="decision-quality candidate arms: 'registered' "
                                "(default: native/damping/pi/wall) or 'e2' "
                                "(+ fixed/switching mix, wall-pilot and "
                                "trajectory arms from evalh.scorer.arm_contrasts)")
            p.add_argument("--extra-cal-labels", dest="extra_cal_labels",
                           action="append", default=None, metavar="PATH",
                           help="repeatable sealed CAL-tune label pack directory "
                                "(SHA256SUMS-verified) carrying the E2 arms; "
                                "every (world, arm) must be a registered E2 arm "
                                "on a released CAL-tune world")
            p.add_argument("--extra-cal-pack", dest="extra_cal_pack",
                           default=None, metavar="PATH",
                           help="sealed CAL-tune FEATURE pack directory for the "
                                "E4 trajectory arms (SHA256SUMS-verified, "
                                "requires --extra-cal-pack-sha256); its arrays "
                                "supply the features the released CAL shards "
                                "do not carry")
            p.add_argument("--extra-cal-pack-sha256",
                           dest="extra_cal_pack_sha256", default=None,
                           help="SHA-256 pin of the sealed feature pack's "
                                "SHA256SUMS (required with --extra-cal-pack)")
    return parser


def main(argv=None, **inject) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "score":
            if getattr(args, "comparators", None) is None:
                raise ScorerCliRefusal(
                    "--comparators is required (comma list from "
                    f"{list(_COMPARATORS)}); no default. Rule-based v2g is "
                    "registered later and is refused in this run.")
            result = run_score(args, **inject)
        else:
            result = run_dry_run(args, source_factory=inject.get(
                "source_factory"))
    except ScorerCliRefusal as exc:
        print(f"[wm-n1 scorer refused] {exc}", file=sys.stderr)
        return 87
    except Exception as exc:                             # noqa: BLE001
        print(f"[wm-n1 scorer refused] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 87
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
