"""Bounded TRAIN-only timestamp audit of the existing label sidecar.

The caller supplies an externally registered raw SHA-256 for a spec naming
exactly one PLAN_RECEIPT, one TRAIN_ISOLATION_RECEIPT and its TRAIN-only
parquet.  The sole parquet projection is :data:`PROJECTION`; physical outcome
values, mixed-pack arrays, original shards and checkpoints are never opened.
This audit describes the frozen mapper's last-row-per-tau behavior; it does
not change labels, train a model, or authorize CAL access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "wm_n1_label_time_audit/1"
PLAN_SCHEMA = "wm_n1_trainstage_plan/1"
RECEIPT_SCHEMA = "wm_n1_train_outcomes_sidecar/1"
RECEIPT_SCHEMA_V2 = "wm_n1_train_outcomes_sidecar/2"
PROJECTION = ("_global_row_id", "world", "source", "chunk_id", "split",
              "t0_s", "policy", "t", "tau", "tau_available")
SOURCE_REL = ("train/dataset.py", "train/label_isolation.py",
              "train/idea_a.py", "corpus/pack.py",
              "features/featurizer.py", "features/outcomes.py")
INPUT_ROLES = ("plan_receipt", "isolation_receipt", "train_labels")
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
N_ITEMS = 2153
N_ROWS = 75624
N_POSITIVE = 1096
N_ZERO = 1057
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

POLICY_CORRESPONDENCE = {
    "damping": "damping_fleet",
    "native": "native_fleet",
    "pi": "pi_fleet",
    "fixed": "fixed_mix",
    "switching": "switching_mix",
}
"""Receipt/run policy -> frozen v0.4a label policy class (exhaustive).

Paths are relative to ``output/wd01``.  Common to every entry: the label
row's ``policy`` column is the source spec's class
(``partB_v0data/build_v04a.py:195``), and the receipt/item ``policy`` is
the inventory/run policy carried into pack items
(``wm_n1/corpus/pack.py:96-99`` and ``:269``).  The verify kinds are
``partB_v0data/actions_v04.py:70-71`` (``ARM_KINDS``).

* ``damping`` <-> ``damping_fleet``: base sources get class
  ``damping_fleet`` with verify kind ``damping_fleet``
  (``partB_v0data/v04a_sources.py:148-149, 166-167``); that kind verifies
  the run's fleet policy / schedule action ``"damping"`` via
  ``kind.split("_")[0]`` (``partB_v0data/actions_v04.py:326-332,
  344-349``).
* ``native`` <-> ``native_fleet``: native_base / native sources get class
  ``native_fleet`` with kind ``native_fleet``
  (``partB_v0data/v04a_sources.py:151-152, 155, 170-171``); the kind
  verifies ``"native"`` (``partB_v0data/actions_v04.py:326-332,
  344-349``).
* ``pi`` <-> ``pi_fleet``: pi sources get class ``pi_fleet`` with kind
  ``pi_fleet`` (``partB_v0data/v04a_sources.py:156, 174-175``); the kind
  verifies schedule actions ``{"pi"}``
  (``partB_v0data/actions_v04.py:344-349``).
* ``fixed`` <-> ``fixed_mix``: fixed1/fixed2 sources get class
  ``fixed_mix`` with verify kind ``"fixed"``
  (``partB_v0data/v04a_sources.py:157-158``), verified as a constant
  per-CAV assignment (``partB_v0data/actions_v04.py:350-354``).
* ``switching`` <-> ``switching_mix``: switching1..3 / swnative / swpi
  sources get class ``switching_mix`` with verify kind ``"switching"``
  (``partB_v0data/v04a_sources.py:159-161, 179-181``), verified as a
  multi-event per-CAV schedule (``partB_v0data/actions_v04.py:355-358``).

No other pair is source-supported; in particular ``wall`` has no cited
label class.  Any pair outside this table refuses (fail closed).
"""


class AuditRefusal(RuntimeError):
    """The registered TRAIN-only access contract is not proved."""


def policy_correspondence(receipt_policy, label_policy):
    """Classify a raw (receipt, label) policy pair; never normalizes names.

    Returns ``"mapped"`` only for an exact cited table entry,
    ``"contradicting"`` when the receipt policy is cited but maps to a
    different label class (or the label class is cited for a different
    receipt policy), else ``"unmapped"``.
    """
    if not (isinstance(receipt_policy, str) and isinstance(label_policy, str)):
        return "unmapped"
    want = POLICY_CORRESPONDENCE.get(receipt_policy)
    if want is not None and want == label_policy:
        return "mapped"
    if want is not None or label_policy in POLICY_CORRESPONDENCE.values():
        return "contradicting"
    return "unmapped"


#: Chunk-4 branch arms stored as their own source name
#: (``partB_v0data/v04a_sources.py:151-164``).
CHUNK4_SOURCES = ("native_base", "native", "pi", "fixed1", "fixed2",
                  "switching1", "switching2", "switching3")
#: Batches whose campaign ``__native``/``__pi`` runs are the switching arms
#: (``partB_v0data/v04a_sources.py:178-182``).
CAMPAIGN_SWITCH_BATCHES = ("chunk1", "chunk1_fix27", "chunk2", "chunk3")
#: Batches holding the separate J3 TRUE fleet arms
#: (``partB_v0data/v04a_sources.py:168-177``).
TRUEARM_BATCHES = ("truearms_chunk2", "truearms_chunk3")


def resolved_source(batch, chunk, arm, policy):
    """Resolve a pack item to its v0.4a label source suffix (or ``None``).

    Mirrors ``features.outcomes.label_source`` (module ``features/outcomes.py``
    ``:87-120``), itself derived verbatim from the v0.4a source registry
    ``partB_v0data/v04a_sources.py:145-182``:

    * ``arm == 'base'`` -> ``'base'`` (every chunk; ``:146, 166-167``);
    * ``chunk == 'chunk4'`` -> the arm as-is for the chunk-4 branch arms
      ``native_base / native / pi / fixed1 / fixed2 / switching1..3``
      (``:147-164``), else ``None`` (e.g. wall arms);
    * ``batch in truearms_chunk2/3`` with arm ``native``/``pi`` -> the arm
      (the J3 TRUE fleet arms; ``:168-177``);
    * ``batch in chunk1 / chunk1_fix27 / chunk2 / chunk3`` with arm
      ``native``/``pi`` and policy ``switching`` -> ``'swnative'``/``'swpi'``
      (the campaign per-CAV switching arms; ``:178-182``);
    * anything else (including the chunk-2/3/1 ``fixed*`` and
      ``switching1..3`` arms, and every wall arm) -> ``None``: no v0.4a
      labels, zero-label.
    """
    if arm is None:
        return None
    arm = str(arm)
    batch = None if batch is None else str(batch)
    chunk = None if chunk is None else str(chunk)
    policy = None if policy is None else str(policy)
    if arm == "base":
        return "base"
    if chunk == "chunk4":
        return arm if arm in CHUNK4_SOURCES else None
    if batch in TRUEARM_BATCHES and arm in ("native", "pi"):
        return arm
    if batch in CAMPAIGN_SWITCH_BATCHES and arm in ("native", "pi") \
            and policy == "switching":
        return "swnative" if arm == "native" else "swpi"
    return None


def require(ok, message):
    if not ok:
        raise AuditRefusal(message)


def sha_pin(value):
    require(isinstance(value, str) and _SHA_RE.fullmatch(value) is not None,
            "exact lowercase SHA-256 pin required")
    return value


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_pinned_json(path, sha):
    pin = sha_pin(sha)
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin, "pinned JSON SHA mismatch")
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), "pinned JSON must be an object")
    return value


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def exact_path(value):
    require(isinstance(value, str) and value and value == value.strip(),
            "absolute literal path required")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts,
            "absolute traversal-free path required")
    return path


def _unsafe_link(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise AuditRefusal(f"cannot inspect path ancestry: {path}") from exc
    return (path.is_symlink()
            or bool(getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label} must be a finite number")
    return float(value)


def exact_int(value, label):
    require(type(value) is int and value >= 0,
            f"{label} must be a nonnegative integer")
    return value


def load_spec(spec_path, spec_sha):
    """Source/role/path restrictions before any registered input hash."""
    spec = read_pinned_json(exact_path(spec_path), spec_sha)
    require(spec.get("schema") == SCHEMA, "unknown audit spec schema")
    require(spec.get("id") == "WM-N1-LABEL-TIME-20261003",
            "unknown audit allocation ID")
    require(isinstance(spec.get("direct_owner_approval"), str)
            and "TRAIN-only label-timing audit" in spec["direct_owner_approval"],
            "direct TRAIN-only owner approval metadata required")
    require(spec.get("checkpoint_lock") ==
            "b37a25a7673cf3ae16effd9690ffba0a027c9763c5695552210e437b7dde2585",
            "checkpoint lock metadata drift")
    require(spec.get("allowed_columns") == list(PROJECTION),
            "spec must pin the sole ten-column projection")
    require(all(spec.get(key) == value for key, value in (
        ("expected_items", N_ITEMS), ("expected_rows", N_ROWS),
        ("expected_positive", N_POSITIVE), ("expected_zero", N_ZERO))),
            "spec must pin exact TRAIN item/row budgets")
    require(spec.get("taus") == list(TAUS), "spec taus differ from frozen model")
    sha_pin(spec.get("expected_pack_manifest_sha256"))
    sha_pin(spec.get("expected_pack_index_sha256"))
    root = Path(__file__).resolve().parents[1]
    sources = spec.get("source_files")
    require(isinstance(sources, list) and len(sources) == len(SOURCE_REL),
            "exact reviewed source hash set required")
    for entry, rel in zip(sources, SOURCE_REL):
        require(isinstance(entry, dict) and set(entry) == {"path", "sha256"},
                "source entry needs exact path/SHA fields")
        path = exact_path(entry["path"])
        require(path.resolve(strict=True) == root.joinpath(*rel.split("/")),
                f"reviewed source path mismatch for {rel}")
        sha_pin(entry["sha256"])
    inputs = spec.get("inputs")
    require(isinstance(inputs, dict) and set(inputs) == set(INPUT_ROLES),
            "exact TRAIN input role set required")
    for role in INPUT_ROLES:
        rec = inputs[role]
        require(isinstance(rec, dict) and set(rec) == {"path", "sha256"},
                f"{role} needs exact path/SHA fields")
        path = exact_path(rec["path"])
        require(not any(_unsafe_link(part) for part in (path, *path.parents)),
                f"{role} path traverses a symlink or reparse point")
        require(path.drive.lower() == "e:"
                or (path.drive.lower() == "d:"
                    and path.as_posix().lower().startswith("d:/rd-staging/")),
                "registered input outside approved E: or D:/rd-staging")
        want_name = {"plan_receipt": "PLAN_RECEIPT.json",
                     "isolation_receipt": "TRAIN_ISOLATION_RECEIPT.json",
                     "train_labels": "train_outcomes.parquet"}[role]
        require(path.name == want_name, f"unexpected {role} filename")
        sha_pin(rec["sha256"])
    out = exact_path(spec.get("output_dir"))
    require(out.drive.lower() == "e:" and not out.exists(),
            "output must be a fresh E: directory")
    require(out.parent.is_dir(), "fresh output parent does not exist")
    require(not any(_unsafe_link(parent) for parent in (out, *out.parents)),
            "output path traverses a symlink or junction")
    resolved_parent = out.parent.resolve(strict=True)
    protected = [root, *(Path(inputs[role]["path"]).parent.resolve(strict=True)
                         for role in INPUT_ROLES)]
    require(all(resolved_parent != tree and tree not in resolved_parent.parents
                for tree in protected),
            "output parent/source overlap")
    return spec


def verify_sources(spec):
    for entry, rel in zip(spec["source_files"], SOURCE_REL):
        path = Path(entry["path"])
        require(path.is_file() and not path.is_symlink(),
                f"reviewed source unavailable: {rel}")
        require(hash_file(path) == entry["sha256"],
                f"reviewed source drift: {rel}")


def _plan_selection(plan_receipt):
    require(plan_receipt.get("schema") == PLAN_SCHEMA,
            "unknown plan receipt schema")
    plan = plan_receipt.get("plan")
    require(isinstance(plan, dict) and isinstance(plan.get("selection"), list),
            "plan selection missing")
    require(plan_receipt.get("selection_sha256") == canonical_sha(plan),
            "plan selection canonical SHA mismatch")
    require(exact_int(plan.get("k_max"), "K") == 192,
            "audit is registered for K=192")
    return plan, plan["selection"]


def validate_receipts(plan_receipt, isolation, spec):
    """Map every approved global label row to its TRAIN item using metadata."""
    plan, selection = _plan_selection(plan_receipt)
    is_v2 = isolation.get("schema") == RECEIPT_SCHEMA_V2
    require(isolation.get("schema") in (RECEIPT_SCHEMA, RECEIPT_SCHEMA_V2)
            and isolation.get("split") == "TRAIN",
            "not a schema/1 or schema/2 TRAIN isolation receipt")
    require(len(selection) == N_ITEMS, "PLAN item count exceeds registered budget")
    require(plan.get("n_items") >= len(selection), "PLAN item count invalid")
    pack = isolation.get("pack") or {}
    require(pack.get("dir") == plan_receipt.get("pack")
            and pack.get("manifest_sha256") == plan.get("manifest_sha256")
            == spec["expected_pack_manifest_sha256"]
            and pack.get("index_sha256") == plan.get("index_sha256")
            == spec["expected_pack_index_sha256"],
            "plan/isolation pack source hashes disagree")
    sidecar = isolation.get("sidecar") or {}
    require(sidecar.get("file") == "train_outcomes.parquet"
            and sidecar.get("sha256") == spec["inputs"]["train_labels"]["sha256"]
            and sidecar.get("parquet_index") == "_global_row_id",
            "TRAIN sidecar identity/count differs from spec")
    columns = sidecar.get("columns")
    require(isinstance(columns, dict) and set(PROJECTION) <= set(columns),
            "TRAIN sidecar lacks one of the ten allowed columns")
    runs, items = isolation.get("runs"), isolation.get("items")
    require(isinstance(runs, list) and isinstance(items, list)
            and len(runs) == len(items) == N_ITEMS,
            "isolation run/item count mismatch")
    id_to_item = {}
    positive = zero = 0
    expected_rows_v2 = 0
    ordered = []
    seen_keys = set()
    last_position = -1
    for plan_item, run, item in zip(selection, runs, items):
        require(all(isinstance(x, dict) for x in (plan_item, run, item)),
                "invalid plan/run/item entry")
        key = (f'{plan_item.get("batch")}/{plan_item.get("world")}__'
               f'{plan_item.get("arm")}')
        require(key == plan_item.get("key") == run.get("key")
                and key not in seen_keys, "TRAIN run key mismatch/duplicate")
        seen_keys.add(key)
        position = exact_int(plan_item.get("position"), "plan position")
        require(position > last_position, "plan positions duplicate/out of order")
        last_position = position
        for field in ("world", "arm", "chunk", "batch", "split", "t0_s",
                      "tick", "plan_row"):
            require(item.get(field) == plan_item.get(field),
                    f"TRAIN item {field} conflicts with plan")
        require(plan_item["split"] == run.get("split") == "TRAIN",
                "non-TRAIN plan/receipt item")
        require(run.get("shard_rel") == plan_item.get("shard_rel")
                and run.get("input_sha256") == plan_item.get("input_sha256"),
                "TRAIN source identity differs from plan")
        n = exact_int(plan_item.get("n_outcome_rows"), "plan label count")
        require(n == exact_int(run.get("n_outcome_rows"), "run label count")
                == exact_int(item.get("n_rows"), "item label count"),
                "TRAIN label row count mismatch")
        first = exact_int(item.get("first_global_row"), "item global first")
        require(first == run.get("first_global_row"),
                "TRAIN run/item row offset mismatch")
        require(isinstance(item.get("policy"), str) and item["policy"],
                "TRAIN policy missing from receipt")
        require(item.get("pair_key") ==
                f'{item["world"]}@t0={float(item["t0_s"]):g}',
                "TRAIN pair key mismatch")
        snapshot = ((0.0 if item["arm"] in ("base", "native_base")
                     else finite(item["t0_s"], "item t0"))
                    + (exact_int(item["tick"], "item tick")
                       - exact_int(plan_item["tick_start"], "run tick start")) * 0.5)
        require(plan_item["tick_start"] <= item["tick"] < plan_item["tick_end"],
                "snapshot tick outside run")
        descriptor = dict(key=key, position=position,
                          world=item["world"], arm=item["arm"],
                          chunk=item["chunk"], policy=item["policy"],
                          batch=item.get("batch", item["chunk"]),
                          t0_s=float(item["t0_s"]),
                          inferred_snapshot_t_s=snapshot,
                          first_global_row=first, n_rows=n)
        ordered.append(descriptor)
        excluded = run.get("excluded") is True
        rebound = run.get("rebound") is True
        require(not (excluded or rebound) or is_v2,
                "excluded/rebound TRAIN run markers require schema/2")
        if n == 0 or excluded:
            zero += 1
            require(run.get("outcomes_sha256") is None,
                    "zero-row/excluded TRAIN run falsely records outcomes")
        else:
            positive += 1
            if rebound:
                require(run.get("outcomes_sha256") is None,
                        "rebound TRAIN run must not claim the shard outcome SHA")
                sha_pin(run.get("label_source_sha256"))
            else:
                sha_pin(run.get("outcomes_sha256"))
            expected_rows_v2 += n
            for row_id in range(first, first + n):
                require(row_id not in id_to_item,
                        "overlapping TRAIN global row IDs")
                id_to_item[row_id] = descriptor
    if is_v2:
        require(sidecar.get("n_rows") == len(id_to_item)
                and expected_rows_v2 == len(id_to_item),
                "schema/2 TRAIN sidecar rows differ from non-excluded coverage")
    else:
        require(sidecar.get("n_rows") == N_ROWS,
                "TRAIN sidecar identity/count differs from spec")
        require((positive, zero, len(id_to_item)) ==
                (N_POSITIVE, N_ZERO, N_ROWS),
                "TRAIN positive/zero/row budgets differ from registered "
                "values")
    selection_meta = isolation.get("selection") or {}
    require(selection_meta.get("n_runs") == N_ITEMS
            and selection_meta.get("n_rows") == N_ROWS,
            "isolation selection totals disagree")
    return ordered, id_to_item


def availability(value, tau):
    """Frozen mapper mask is ``True if None else bool(value)``.

    Numeric ``tau_available`` may encode a duration; report that raw duration
    and whether it reaches tau.  The audit does not reinterpret the training
    mask as full-horizon availability.
    """
    if value is None:
        return dict(kind="null", raw=None, training_mask=True,
                    duration_ge_tau=None)
    if type(value) is bool:
        return dict(kind="boolean", raw=value, training_mask=value,
                    duration_ge_tau=None)
    duration = finite(value, "tau_available")
    return dict(kind="numeric_duration", raw=duration,
                training_mask=bool(duration),
                duration_ge_tau=tau <= duration + 1e-9)


def analyze_projected_rows(projected, ordered, id_to_item):
    """Pure ten-column analysis in original global-row order."""
    require(set(projected) == set(PROJECTION), "unexpected projection fields")
    values = [projected[name] for name in PROJECTION]
    expected_rows = len(id_to_item)
    require(all(len(column) == expected_rows for column in values),
            "projected row count differs from approved item coverage")
    records = []
    seen_ids = set()
    previous_id = -1
    groups = defaultdict(list)
    last_time_tau = {}
    item_order_inversions = defaultdict(int)
    label_policies = defaultdict(set)
    for rec in zip(*values):
        row = dict(zip(PROJECTION, rec))
        row_id = exact_int(row["_global_row_id"], "global row ID")
        require(row_id > previous_id, "sidecar rows changed original ID order")
        previous_id = row_id
        require(row_id not in seen_ids and row_id in id_to_item,
                "sidecar global row ID missing/foreign/duplicated")
        seen_ids.add(row_id)
        item = id_to_item[row_id]
        expected_source = resolved_source(
            item.get("batch", item["key"].split("/", 1)[0]),
            item["chunk"], item["arm"], item["policy"]) or item["arm"]
        require(row["split"] == "TRAIN"
                and row["world"] == item["world"]
                and row["source"] == expected_source
                and row["chunk_id"] == item["chunk"]
                and policy_correspondence(item["policy"],
                                          row["policy"]) == "mapped"
                and finite(row["t0_s"], "label t0") == item["t0_s"],
                "projected label identity differs from approved TRAIN item")
        decision = finite(row["t"], "decision timestamp")
        tau = finite(row["tau"], "outcome tau")
        require(tau in TAUS, "unregistered outcome tau")
        avail = availability(row["tau_available"], tau)
        last_pair = last_time_tau.get(item["position"])
        if last_pair is not None and (decision, tau) < last_pair:
            item_order_inversions[item["position"]] += 1
        last_time_tau[item["position"]] = (decision, tau)
        label_policies[item["position"]].add(row["policy"])
        groups[(item["position"], tau)].append((row_id, decision, avail))
    require(seen_ids == set(id_to_item),
            "projected global IDs do not cover approved TRAIN intervals")
    for item in ordered:
        for tau in TAUS:
            group = groups.get((item["position"], tau), [])
            times = [r[1] for r in group]
            chosen = group[-1] if group else None
            availability_last = chosen[2] if chosen else None
            records.append(dict(
                item_position=item["position"], run_key=item["key"],
                world=item["world"], arm=item["arm"], policy=item["policy"],
                label_policy=sorted(label_policies[item["position"]]),
                chunk=item["chunk"], tau_s=tau, n_rows=len(group),
                has_row_at_declared_t0=any(t == item["t0_s"] for t in times),
                n_rows_duration_lt_tau=sum(
                    r[2]["duration_ge_tau"] is False for r in group),
                n_distinct_t=len(set(times)),
                n_duplicate_same_t=len(group) - len(set(times)),
                item_t_tau_order_inversions=item_order_inversions[item["position"]],
                t_in_original_order_s=times,
                first_global_row_id=group[0][0] if group else None,
                selected_global_row_id=chosen[0] if chosen else None,
                min_t_s=min(times) if times else None,
                max_t_s=max(times) if times else None,
                selected_t_s=chosen[1] if chosen else None,
                declared_t0_s=item["t0_s"],
                inferred_snapshot_t_s=item["inferred_snapshot_t_s"],
                selected_minus_declared_t0_s=(chosen[1] - item["t0_s"]
                                               if chosen else None),
                selected_minus_inferred_snapshot_s=(
                    chosen[1] - item["inferred_snapshot_t_s"]
                    if chosen else None),
                training_mask=(availability_last["training_mask"]
                               if chosen else False),
                availability_kind=(availability_last["kind"]
                                   if chosen else "absent"),
                availability_raw=(availability_last["raw"]
                                  if chosen else None),
                duration_ge_tau=(availability_last["duration_ge_tau"]
                                 if chosen else None)))
    return records


def timing_summary(records):
    """Aggregate :func:`analyze_projected_rows` records (pure, metadata only).

    A cell (item, tau) is *misbound* when the frozen mapper's last row per tau
    has a decision time different from the item's declared ``t0_s``.
    """
    def _count(rows, key):
        out = defaultdict(lambda: dict(cells=0, items=set(), rows=0))
        for r in rows:
            k = key(r)
            out[k]["cells"] += 1
            out[k]["items"].add(r["item_position"])
            out[k]["rows"] += r["n_rows"]
        return {str(k): dict(cells=v["cells"], items=len(v["items"]),
                             rows=v["rows"])
                for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}

    by_item = defaultdict(list)
    for r in records:
        by_item[r["item_position"]].append(r)
    zero_items = [p for p, rs in by_item.items()
                  if sum(r["n_rows"] for r in rs) == 0]
    distinct_t_hist = defaultdict(int)
    rows_hist = defaultdict(int)
    for p, rs in by_item.items():
        times = {t for r in rs for t in r["t_in_original_order_s"]}
        if times:
            distinct_t_hist[len(times)] += 1
            rows_hist[sum(r["n_rows"] for r in rs)] += 1
    rows_t0_tau = defaultdict(int)
    for r in records:
        rows_t0_tau[f'{r["declared_t0_s"]:g}/{r["tau_s"]:g}'] += r["n_rows"]
    present = [r for r in records if r["selected_t_s"] is not None]
    mis = [r for r in present
           if r["selected_t_s"] != r["declared_t0_s"]]
    offsets = defaultdict(int)
    for r in mis:
        offsets[f'{r["selected_minus_declared_t0_s"]:g}'] += 1
    def batch(r):
        return r["run_key"].split("/", 1)[0]
    return dict(
        n_items=len(by_item), n_zero_label_items=len(zero_items),
        n_positive_items=len(by_item) - len(zero_items),
        n_rows=sum(r["n_rows"] for r in records),
        items_by_distinct_decision_times={
            str(k): v for k, v in sorted(distinct_t_hist.items())},
        items_by_label_row_count={
            str(k): v for k, v in sorted(rows_hist.items())},
        rows_by_t0_tau=dict(sorted(rows_t0_tau.items())),
        n_present_cells=len(present),
        n_cells_multiple_decision_times=sum(r["n_distinct_t"] > 1
                                            for r in present),
        n_cells_duplicate_same_t=sum(r["n_duplicate_same_t"] > 0
                                     for r in present),
        n_rows_duration_lt_tau=sum(r["n_rows_duration_lt_tau"]
                                   for r in records),
        n_cells_selected_mask_false=sum(not r["training_mask"]
                                        for r in present),
        misbinding=dict(
            n_cells=len(mis),
            n_items=len({r["item_position"] for r in mis}),
            n_rows_in_misbound_cells=sum(r["n_rows"] for r in mis),
            n_cells_t0_row_present_but_overridden=sum(
                r["has_row_at_declared_t0"] for r in mis),
            n_cells_no_t0_row=sum(not r["has_row_at_declared_t0"]
                                  for r in mis),
            selected_minus_t0_s=dict(sorted(offsets.items())),
            by_chunk_batch_arm_policy=_count(mis, lambda r: (
                r["chunk"], batch(r), r["arm"], r["policy"],
                "|".join(r["label_policy"]))),
            by_tau=_count(mis, lambda r: r["tau_s"])),
        present_by_chunk_batch_arm_policy=_count(present, lambda r: (
            r["chunk"], batch(r), r["arm"], r["policy"],
            "|".join(r["label_policy"]))),
        n_cells_selected_ne_inferred_snapshot=sum(
            r["selected_minus_inferred_snapshot_s"] != 0 for r in present))


def _read_projection(path, expected_rows=None):
    """The sole data-body read: exact ten columns, single-threaded Arrow.

    ``expected_rows`` defaults to the module registered budget :data:`N_ROWS`
    (schema/1); the schema/2 sidecar path passes its own receipt row count.
    """
    import pyarrow.parquet as pq

    if expected_rows is None:
        expected_rows = N_ROWS
    require(type(expected_rows) is int and expected_rows >= 0,
            "expected row count must be a nonnegative integer")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == expected_rows,
            "parquet footer row count exceeds or differs from budget")
    require(set(PROJECTION) <= set(parquet.schema_arrow.names),
            "parquet lacks approved projection column")
    table = parquet.read(columns=list(PROJECTION), use_threads=False)
    require(table.num_rows == expected_rows
            and table.column_names == list(PROJECTION),
            "Arrow returned unexpected projection shape")
    return {name: table.column(name).to_pylist() for name in PROJECTION}


def _write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_audit(spec_path, spec_sha):
    spec = load_spec(spec_path, spec_sha)
    out = Path(spec["output_dir"])
    out.mkdir(exist_ok=False)
    try:
        verify_sources(spec)
        inputs = spec["inputs"]
        plan = read_pinned_json(inputs["plan_receipt"]["path"],
                                inputs["plan_receipt"]["sha256"])
        receipt = read_pinned_json(inputs["isolation_receipt"]["path"],
                                   inputs["isolation_receipt"]["sha256"])
        ordered, id_to_item = validate_receipts(plan, receipt, spec)
        labels = Path(inputs["train_labels"]["path"])
        require(hash_file(labels) == inputs["train_labels"]["sha256"],
                "TRAIN-only label sidecar SHA mismatch")
        projected = _read_projection(labels)
        require(hash_file(labels) == inputs["train_labels"]["sha256"],
                "TRAIN-only label sidecar changed during projected read")
        records = analyze_projected_rows(projected, ordered, id_to_item)
        verify_sources(spec)
        fields = list(records[0])
        with (out / "ITEM_TAU_TIMES.csv").open("x", newline="",
                                                encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        _write_new_json(out / "ITEM_TAU_TIMES.json", dict(
            schema="wm_n1_train_item_tau_times/1", rows=records))
        inverted_positions = {r["item_position"] for r in records
                              if r["item_t_tau_order_inversions"] > 0}
        by_tau = {}
        for tau in TAUS:
            cells = [r for r in records if r["tau_s"] == tau]
            by_tau[str(int(tau))] = dict(
                present=sum(r["n_rows"] > 0 for r in cells),
                absent=sum(r["n_rows"] == 0 for r in cells),
                distinct_t_gt1=sum(r["n_distinct_t"] > 1 for r in cells),
                selected_ne_declared_t0=sum(
                    r["selected_t_s"] is not None
                    and r["selected_minus_declared_t0_s"] != 0 for r in cells),
                selected_ne_inferred_snapshot=sum(
                    r["selected_t_s"] is not None
                    and r["selected_minus_inferred_snapshot_s"] != 0
                    for r in cells),
                training_mask_true=sum(r["training_mask"] for r in cells),
                numeric_duration_lt_tau=sum(
                    r["availability_kind"] == "numeric_duration"
                    and r["duration_ge_tau"] is False for r in cells))
        summary = dict(
            schema=SCHEMA, allocation_id=spec["id"],
            input_sha256={role: inputs[role]["sha256"] for role in INPUT_ROLES},
            n_items=N_ITEMS, n_rows=N_ROWS, n_positive=N_POSITIVE, n_zero=N_ZERO,
            n_item_tau=len(records),
            n_duplicate_t_cells=sum(r["n_distinct_t"] > 1 for r in records),
            n_duplicate_same_t_cells=sum(r["n_duplicate_same_t"] > 0
                                         for r in records),
            n_items_t_tau_order_inverted=len(inverted_positions),
            by_tau=by_tau,
            n_selected_t_ne_declared_t0=sum(
                r["selected_t_s"] is not None
                and r["selected_minus_declared_t0_s"] != 0 for r in records),
            n_selected_t_ne_inferred_snapshot=sum(
                r["selected_t_s"] is not None
                and r["selected_minus_inferred_snapshot_s"] != 0
                for r in records),
            snapshot_inference="features.featurizer: base/native_base origin 0; "
                               "other arms origin=t0; tick offset * 0.5 s",
            selected_timestamp_rule="train.dataset.map_outcome_targets: "
                                    "last row per tau in original row-ID order",
            availability_duration_rule="features v0.4a producer: "
                                       "tau <= tau_available + 1e-9; "
                                       "training mask remains bool(value), "
                                       "with null mapped to true",
            projection=list(PROJECTION),
            item_tau_csv="ITEM_TAU_TIMES.csv",
            item_tau_json="ITEM_TAU_TIMES.json")
        verify_sources(spec)
        _write_new_json(out / "REPORT.json", summary)
        return summary
    except Exception as exc:
        _write_new_json(out / "FAILURE.json", dict(
            schema="wm_n1_train_label_time_audit_failure/1",
            allocation_id=spec["id"],
            error_type=type(exc).__name__, error=str(exc)))
        raise


def _load_plan_sidecar(pack_dir):
    """Pack-derived TRAIN plan (metadata only); imports the isolation module."""
    try:
        from . import label_isolation as isolation
    except ImportError:  # loaded as a top-level module (tests)
        from train import label_isolation as isolation
    return isolation.plan_sidecar(pack_dir, None)


def run_audit_on_sidecar(sidecar_dir, pack_dir, out_dir, *,
                         input_receipt_sha256=None,
                         input_sidecar_sha256=None):
    """Audit a schema/1 or schema/2 TRAIN sidecar from its own receipt.

    Unlike :func:`run_audit` this needs no separate registered spec: it reads
    the sidecar's own ``TRAIN_ISOLATION_RECEIPT.json`` and
    ``train_outcomes.parquet`` plus only the pack metadata (via
    ``plan_sidecar``), validates the ten-column projection, applies the exact
    schema/2 rebound/excluded contract of :func:`validate_receipts`, and runs
    the same :func:`timing_summary` aggregation, writing ``ITEM_TAU_TIMES.json``
    and ``REPORT.json`` into a fresh ``out_dir``.  Nothing here is run against
    the real sidecar by this change; it is the code path the coordinator runs
    once later.
    """
    sidecar_dir = Path(sidecar_dir)
    pack_dir = Path(pack_dir)
    out = exact_path(out_dir)
    require(not out.exists(), "audit output must be a fresh directory")
    receipt_path = sidecar_dir / "TRAIN_ISOLATION_RECEIPT.json"
    parquet_path = sidecar_dir / "train_outcomes.parquet"
    receipt_sha = hash_file(receipt_path)
    sidecar_sha = hash_file(parquet_path)
    if input_receipt_sha256 is not None:
        require(receipt_sha == sha_pin(input_receipt_sha256),
                "input receipt SHA256 does not match the pin")
    if input_sidecar_sha256 is not None:
        require(sidecar_sha == sha_pin(input_sidecar_sha256),
                "input sidecar parquet SHA256 does not match the pin")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(isinstance(receipt, dict)
            and receipt.get("schema") in (RECEIPT_SCHEMA, RECEIPT_SCHEMA_V2)
            and receipt.get("split") == "TRAIN",
            "not a schema/1 or schema/2 TRAIN isolation receipt")
    pack = receipt.get("pack") or {}
    require(hash_file(pack_dir / "manifest.json") == pack.get("manifest_sha256")
            and hash_file(pack_dir / "INDEX.json") == pack.get("index_sha256"),
            "pack metadata SHA differs from the sidecar receipt")
    plan = _load_plan_sidecar(pack_dir)
    runs, items = receipt.get("runs") or [], receipt.get("items") or []
    plan_runs = {run["key"]: run for run in plan["train_runs"]}
    require(len(runs) == len(items) == len(plan_runs),
            "sidecar receipt run/item count differs from the pack plan")
    selection = []
    for position, (run, item) in enumerate(zip(runs, items)):
        plan_run = plan_runs.get(str(run.get("key")))
        require(plan_run is not None,
                "sidecar receipt run is not a pack TRAIN run")
        selection.append(dict(
            position=position, key=run["key"], batch=item.get("batch"),
            chunk=item.get("chunk"), world=item["world"], arm=item["arm"],
            split=item["split"], t0_s=item["t0_s"], tick=int(item["tick"]),
            plan_row=int(item["plan_row"]), tick_start=plan_run["tick_start"],
            tick_end=plan_run["tick_end"],
            n_outcome_rows=int(run["n_outcome_rows"]),
            shard_rel=run["shard_rel"], input_sha256=run["input_sha256"]))
    plan_doc = dict(k_max=plan["k_max"], n_items=len(selection),
                    manifest_sha256=pack.get("manifest_sha256"),
                    index_sha256=pack.get("index_sha256"), selection=selection)
    plan_receipt = dict(schema=PLAN_SCHEMA, pack=pack.get("dir"),
                        plan=plan_doc,
                        selection_sha256=canonical_sha(plan_doc))
    spec = dict(id="WM-N1-LABEL-TIME-20261003",
                inputs={"train_labels": {"sha256": sidecar_sha}},
                expected_pack_manifest_sha256=pack.get("manifest_sha256"),
                expected_pack_index_sha256=pack.get("index_sha256"))
    ordered, id_to_item = validate_receipts(plan_receipt, receipt, spec)
    expected_rows = exact_int((receipt.get("sidecar") or {}).get("n_rows"),
                              "sidecar row count")
    require(expected_rows == len(id_to_item),
            "sidecar n_rows differs from non-excluded coverage")
    projected = _read_projection(parquet_path, expected_rows=expected_rows)
    records = analyze_projected_rows(projected, ordered, id_to_item)
    out.mkdir(exist_ok=False)
    summary = dict(
        schema=SCHEMA, allocation_id="WM-N1-LABEL-TIME-20261003",
        receipt_schema=receipt.get("schema"),
        sidecar_dir=str(sidecar_dir), pack_dir=str(pack_dir),
        input_receipt_sha256=receipt_sha,
        input_sidecar_sha256=sidecar_sha,
        n_items=len(ordered), n_rows=len(id_to_item),
        n_positive=sum(1 for run in runs
                       if int(run["n_outcome_rows"]) > 0
                       and run.get("excluded") is not True),
        n_zero=sum(1 for run in runs
                   if int(run["n_outcome_rows"]) == 0
                   or run.get("excluded") is True),
        projection=list(PROJECTION),
        item_tau_json="ITEM_TAU_TIMES.json",
        timing=timing_summary(records))
    _write_new_json(out / "ITEM_TAU_TIMES.json", dict(
        schema="wm_n1_train_item_tau_times/1", rows=records))
    _write_new_json(out / "REPORT.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--sidecar")
    parser.add_argument("--pack")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.sidecar or args.pack or args.out:
        require(bool(args.sidecar and args.pack and args.out),
                "--sidecar, --pack and --out must be given together")
        summary = run_audit_on_sidecar(args.sidecar, args.pack, args.out)
    else:
        require(bool(args.spec and args.spec_sha256),
                "--spec and --spec-sha256 are required without --sidecar")
        summary = run_audit(args.spec, args.spec_sha256)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
