"""The clean ``wms`` package reproduces the golden artifacts captured from the legacy code.

``tests/golden/data`` was captured from ``wms/r3-int`` (bf7d4cb) by ``capture_golden.py`` (see
README.md).  Every test here rebuilds the same quantity with the NEW stack only
(``wms.data`` / ``wms.model`` / ``wms.train`` / ``wms.eval`` through ``wms.cli.assemble``) and
compares it with the stored artifact using ``golden_compare`` and its tolerance table.

Needs the real local packs (registered in ``configs/data_registry.toml``); skipped when absent.
CPU only, 2 torch threads, ``torch.use_deterministic_algorithms(True)`` as at capture time.
Select parts with ``GOLDEN_PARTS=selection,batches,forward,losses,train,eval``.

Tolerance notes (integration, documented in docs/ARCHITECTURE_V4_CHANGES.md):
* ``selection.json``: absolute path strings inside ``window_filter`` differ by worktree and are
  compared by basename; everything else is exact.
* hash keys (``checksum``/``sha256_float32``/``init_checksum``) are compared only with
  ``STRICT_HASHES=1``; the parameter npz files carry the numeric comparison.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import golden_compare as gc  # noqa: E402

torch = pytest.importorskip("torch")

PARTS = tuple(p for p in os.environ.get(
    "GOLDEN_PARTS", "selection,batches,forward,losses,train,eval").split(",") if p)
STRICT_HASHES = os.environ.get("STRICT_HASHES") == "1"
THREADS = 2
SEED_MODEL, SEED_BATCH_PERM, SEED_ITEM, SEED_LOSS, SEED_TRAIN = 101, 2026, 7000, 900, 1100
N_BATCHES, BATCH_SIZE, HISTORY_S = 3, 2, 60.0
C1_DIR = gc.DATA / "c1"


def _registry_paths():
    from wms.data.registry import resolve

    names = ("pack_v3", "pack_dspeed", "pack_damping", "sidecars_packv3_v2", "identities_dspeed",
             "identities_damping", "overlay_dspeed", "overlay_damping", "lpwo_v3",
             "eval_states_wall_same")
    return {n: resolve(n) for n in names}


try:
    _PATHS = _registry_paths()
    _MISSING = [n for n, p in _PATHS.items() if not Path(p).exists()]
except Exception as exc:  # noqa: BLE001
    _PATHS, _MISSING = {}, [f"registry: {exc}"]

pytestmark = [
    pytest.mark.skipif(not (gc.DATA / "meta.json").exists(), reason="golden data not captured"),
    pytest.mark.skipif(bool(_MISSING), reason=f"local packs not registered/present: {_MISSING}"),
]


def need(part: str) -> None:
    if part not in PARTS:
        pytest.skip(f"part {part} not selected")


# ------------------------------------------------------------------------------ setup
@pytest.fixture(scope="module", autouse=True)
def deterministic():
    torch.set_num_threads(THREADS)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("golden_new")


def r3_config(unroll_s: float, history_s, out_dir):
    """The golden R3 configuration (configs/golden_r3.toml) at one unroll horizon."""
    from wms.config.load import load_raw
    from wms.config.schema import from_mapping
    from wms.config.train import TrainConfig

    raw = load_raw(REPO / "configs" / "golden_r3.toml")
    raw["curriculum"]["unroll_s"] = [float(unroll_s)]
    raw["curriculum"]["stage_steps"] = [3]
    if history_s is None:
        raw["data"].pop("history_s", None)
        raw["data"].pop("history_stride_s", None)
    else:
        raw["data"]["history_s"] = float(history_s)
        raw["data"]["history_stride_s"] = 1.0
    raw["runtime"]["out_dir"] = str(out_dir)
    return from_mapping(TrainConfig, raw)


_BUILT: dict = {}


def built(workdir, unroll_s: float, history_s):
    """(cfg, TrainingInputs) for one variant; datasets are lazy so this is cheap."""
    from wms.cli.assemble import assemble

    key = (float(unroll_s), history_s)
    if key not in _BUILT:
        cfg = r3_config(unroll_s, history_s, workdir / f"run_{int(unroll_s)}_{history_s}")
        with contextlib.redirect_stderr(None):
            _BUILT[key] = (cfg, assemble(cfg, workdir))
    return _BUILT[key]


def np_(t):
    return t.detach().cpu().numpy()


def write_ready(obj):
    """JSON round trip so that numpy/torch scalars compare like the stored file."""
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def sha256_bytes(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def state_checksum(model) -> dict:
    sd = {k: np_(v).astype(np.float32) for k, v in sorted(model.state_dict().items())}
    flat = [v.ravel() for v in sd.values()]
    return {"sha256_float32": sha256_bytes(*flat), "n_tensors": len(sd),
            "n_params": int(sum(v.size for v in flat)),
            "per_tensor_sum": {k: float(v.astype(np.float64).sum()) for k, v in sd.items()},
            "per_tensor_abs_sum": {k: float(np.abs(v.astype(np.float64)).sum())
                                   for k, v in sd.items()}}


def state_npz(model) -> dict:
    return {k.replace(".", "/"): np_(v) for k, v in sorted(model.state_dict().items())}


def scalar_record(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        if k == "seconds":
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            out[k] = [float(x) for x in v]
        elif isinstance(v, dict):
            out[k] = v
    return out


def check_json(actual, name):
    errs = gc.compare_json(write_ready(actual), name, strict_hashes=STRICT_HASHES)
    assert not errs, "\n".join(errs[:20])


def check_json_without_unstored(actual, name, unstored_prefixes):
    """Like :func:`check_json`, ignoring per-tensor sums of weights the checkpoint does not store.

    The C1 checkpoint predates ``driver_encoder.*``: those tensors keep their (unseeded, random)
    construction values in both the legacy and the new loader and are unused in residual mode,
    so their sums are meaningless.  Every stored tensor is still compared at the stored tolerance.
    """
    def drop(block):
        for field in ("per_tensor_sum", "per_tensor_abs_sum"):
            block["state_checksum"][field] = {
                k: v for k, v in block["state_checksum"][field].items()
                if not k.startswith(unstored_prefixes)}
        return block

    expected = drop(gc.load_json(name))
    errs: list[str] = []
    gc._cmp(drop(write_ready(actual)), expected, gc.TOLERANCES.get(name, 0.0), name, errs,
            STRICT_HASHES)
    assert not errs, "\n".join(errs[:20])


def check_npz(actual, name):
    errs = gc.compare_npz(actual, name)
    assert not errs, "\n".join(errs[:20])


# --------------------------------------------------------------------------- selection
def _basenames(node):
    """Replace absolute list paths of a data receipt by their file names."""
    if isinstance(node, dict):
        return {k: (Path(v).name if k == "window_list" and isinstance(v, str)
                    else [Path(x).name for x in v] if k == "exclude_window_list"
                    and isinstance(v, list) else _basenames(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [_basenames(x) for x in node]
    return node


def test_selection_counts_and_keys(workdir):
    need("selection")
    from wms.data.corpus import PackCorpus
    from wms.data.windows import item_key, select_lpo_positions

    cfg, inputs = built(workdir, 120.0, None)       # the true R3 horizon (curriculum max 120 s)
    expected = gc.load_json("selection.json")
    result = {"receipt": {}, "kept_keys": {}}
    for split, ds in (("train", inputs.train), ("val", inputs.val)):
        rc = ds.receipt
        result["receipt"][split] = {k: rc[k] for k in (
            "kept_by_pack", "total_by_pack", "n_records", "n_corpus_windows",
            "window_filter", "mode", "horizon_steps") if k in rc}
        result["receipt"][split]["n_dataset_items"] = len(ds)
    packs = [str(_PATHS[n]) for n in ("pack_v3", "pack_dspeed", "pack_damping")]
    corpus = PackCorpus.open(
        packs[0], extra_pack_dirs=packs[1:],
        extra_overlay_dirs=[str(_PATHS["overlay_dspeed"]), str(_PATHS["overlay_damping"])],
        extra_identity_dirs=[str(_PATHS["identities_dspeed"]), str(_PATHS["identities_damping"])],
        horizon_steps=240)
    names = [Path(p).name for p in packs]
    lists = Path(_PATHS["lpwo_v3"])
    for split, fname in (("train", "windows_train_noPI_noWall.json"),
                         ("val", "windows_val_noPI_noWall.json")):
        selected, _ = select_lpo_positions(
            corpus, names, window_list=str(lists / fname),
            exclude_window_list=[str(lists / "excluded_runs_PI_wall_v3.json")])
        rows = []
        for i in selected:
            rows.append([names[corpus.pack_index(i)], *item_key(corpus.meta(i))])
        result["kept_keys"][split] = rows
        by_pack: dict = {}
        for r in rows:
            by_pack[r[0]] = by_pack.get(r[0], 0) + 1
        result["receipt"][split]["recomputed_kept_by_pack"] = dict(sorted(by_pack.items()))
        assert len(rows) == result["receipt"][split]["n_dataset_items"]
    result["expected_pack_v3"] = {"train": 294, "val": 47}
    assert result["receipt"]["train"]["kept_by_pack"]["pack_v3"] == 294
    assert result["receipt"]["val"]["kept_by_pack"]["pack_v3"] == 47
    assert len(result["kept_keys"]["train"]) == 911 and len(result["kept_keys"]["val"]) == 151
    errs: list[str] = []
    gc._cmp(_basenames(write_ready(result)), _basenames(expected), 0.0, "selection.json", errs,
            STRICT_HASHES)
    assert not errs, "\n".join(errs[:20])


# ------------------------------------------------------------------------------ batches
BATCH_TENSORS = ("states", "present", "slot_index", "loss_valid", "vehicle_ids",
                 "cav_teacher", "is_cav", "hist", "hist_mask", "driver_hist",
                 "driver_hist_mask", "controller_id")
BATCH_FILES = [f"batches_u{u}_hist{'on' if h else 'off'}" for u in (10, 60) for h in (True, False)]


def decode_batches(workdir, unroll_s: int, hist: bool):
    """3 fixed batches of 2 windows (start-shift draws seeded per item), as at capture time."""
    from wms.data.collate import collate

    cfg, inputs = built(workdir, float(unroll_s), HISTORY_S if hist else None)
    ds = inputs.train
    g = torch.Generator().manual_seed(SEED_BATCH_PERM)
    perm = [int(i) for i in torch.randperm(len(ds), generator=g)[:N_BATCHES * BATCH_SIZE]]
    batches, meta = [], {"indices": perm, "items": []}
    for j in range(N_BATCHES):
        items = []
        for n in range(BATCH_SIZE):
            idx = perm[j * BATCH_SIZE + n]
            torch.manual_seed(SEED_ITEM + j * BATCH_SIZE + n)
            items.append(ds[idx])
            meta["items"].append({"dataset_index": idx,
                                  "start_shift_ticks": int(items[-1].get("start_shift_ticks") or 0)})
        batch = collate(items)
        meta.setdefault("identities", []).append(
            [None if x is None else json.loads(json.dumps(x, default=str))
             for x in batch["identities"]])
        batches.append(batch)
    return cfg, inputs, batches, meta


@pytest.mark.parametrize("tag", BATCH_FILES)
def test_decoded_batches(workdir, tag):
    need("batches")
    unroll_s, hist = int(tag.split("_u")[1].split("_")[0]), tag.endswith("histon")
    _, _, batches, meta = decode_batches(workdir, unroll_s, hist)
    arrays, shapes = {}, {}
    for j, batch in enumerate(batches):
        for name in BATCH_TENSORS:
            if name in batch:
                arrays[f"b{j}_{name}"] = np_(batch[name])
                shapes[f"b{j}_{name}"] = [list(batch[name].shape), str(batch[name].dtype)]
    meta.update(unroll_s=unroll_s, history_s=HISTORY_S if hist else None, n_batches=N_BATCHES,
                batch_size=BATCH_SIZE, shapes=shapes,
                seeds={"perm": SEED_BATCH_PERM, "item": SEED_ITEM}, start_shift_prob=0.75)
    check_npz(arrays, f"{tag}.npz")
    check_json(meta, f"{tag}.json")


# ------------------------------------------------------------------------------ forward
def rollout_outputs(model, batch: dict, n_steps: int) -> dict:
    """Teacher-forced rollout with the trainer's argument convention (no grad)."""
    from wms.train.inputs import state_from_kin

    states = batch["states"].float()
    present = batch["present"].bool()
    is_cav = batch["is_cav"].bool()
    teacher = batch["cav_teacher"].float()
    initial = state_from_kin(states[:, 0], present[:, 0])
    initial["is_cav"] = is_cav[:, 0]
    hist, hist_mask = batch.get("hist"), batch.get("hist_mask")
    with torch.no_grad():
        roll = model.rollout(
            initial, n_steps=n_steps, k0=0, cav_teacher=teacher[:, :n_steps + 1],
            present_seq=present[:, :n_steps + 1], is_cav_seq=is_cav[:, :n_steps + 1],
            controller_id=batch.get("controller_id"), history=hist, history_mask=hist_mask,
            driver_history=batch.get("driver_hist", hist),
            driver_history_mask=batch.get("driver_hist_mask", hist_mask))
        gate = model.gate(torch.arange(n_steps))
    out = {"accel_mps2": roll["accel"], "raw_residual_mps2": roll["raw_residual_mps2"],
           "gate": gate, "lane_logits": roll["lane_logits"], "positions": roll["positions"],
           "speeds": roll["speeds"], "lanes": roll["lanes"], "masks": roll["masks"],
           "lane_change_probabilities": roll["lane_change_probabilities"]}
    return {k: np_(v) for k, v in out.items()}


def load_c1():
    from wms.model.checkpoint import load_checkpoint

    receipt = json.loads((C1_DIR / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    return load_checkpoint(C1_DIR / "wm_s.pt", physics=receipt["physics"]).model.eval()


def fresh_r3(workdir, unroll_s: float = 10.0, hist=HISTORY_S):
    """Fresh R3 model (seed 101) + trainer; identical on every call."""
    from wms.cli.assemble import build_model
    from wms.train.trainer import WmSTrainer

    cfg, inputs = built(workdir, unroll_s, hist)
    torch.manual_seed(SEED_MODEL)
    model = build_model(cfg, inputs.physics)
    return WmSTrainer(cfg, model, None, out_dir=workdir / f"fresh_{os.getpid()}")


def test_c1_forward_matches_golden(workdir):
    need("forward")
    c1 = load_c1()
    _, _, off10, _ = decode_batches(workdir, 10, False)
    _, _, off60, _ = decode_batches(workdir, 60, False)
    arrays = {}
    for j in range(N_BATCHES):
        for k, v in rollout_outputs(c1, off10[j], 20).items():
            arrays[f"u10_b{j}_{k}"] = v
    for k, v in rollout_outputs(c1, off60[0], 120).items():
        arrays[f"u60_b0_{k}"] = v
    check_npz(arrays, "forward_c1.npz")
    stored = torch.load(C1_DIR / "wm_s.pt", map_location="cpu", weights_only=True)["model"]
    unstored = tuple(sorted({k.split(".")[0] + "." for k in c1.state_dict() if k not in stored}))
    assert unstored == ("driver_encoder.",), unstored       # exactly what the checkpoint predates
    check_json_without_unstored({
        "checkpoint": "tests/golden/data/c1/wm_s.pt",
        "checkpoint_sha256": hashlib.sha256((C1_DIR / "wm_s.pt").read_bytes()).hexdigest(),
        "flags": "TRAIN_RECEIPT.json config (no history, no soft bound, merge_cal False)",
        "batches": "batches_u10_histoff / batches_u60_histoff", "mode": "eval, no_grad",
        "state_checksum": state_checksum(c1)}, "forward_c1.json", unstored)


def test_r3_init_and_forward_match_golden(workdir):
    need("forward")
    trainer = fresh_r3(workdir)
    model = trainer.model
    check_npz(state_npz(model), "r3_init_state.npz")          # exact: same seed, same init order
    _, _, on10, _ = decode_batches(workdir, 10, True)
    _, _, on60, _ = decode_batches(workdir, 60, True)
    model.eval()
    arrays = {}
    for j in range(N_BATCHES):
        for k, v in rollout_outputs(model, on10[j], 20).items():
            arrays[f"u10_b{j}_{k}"] = v
    for k, v in rollout_outputs(model, on60[0], 120).items():
        arrays[f"u60_b0_{k}"] = v
    check_npz(arrays, "forward_r3.npz")
    check_json({
        "init_seed": SEED_MODEL, "mode": "eval, no_grad", "state_checksum": state_checksum(model),
        "model_attrs": {"residual_soft_bound_mps2": model.residual_soft_bound_mps2,
                        "residual_soft_bound_neg_mps2": model.residual_soft_bound_neg_mps2,
                        "history_s": model.history_s, "rollout_mode": model.rollout_mode,
                        "lane_head": "off" if getattr(model, "lane_change_head", None) is None
                        else "history"}}, "forward_r3.json")


# ------------------------------------------------------------------- losses / 3 steps
def grad_groups(model) -> dict:
    per_param, groups, total = {}, {}, 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            per_param[name] = {"norm": 0.0, "sum": 0.0}
            continue
        g = p.grad.detach().double()
        n2 = float(g.pow(2).sum())
        per_param[name] = {"norm": n2 ** 0.5, "sum": float(g.sum())}
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0.0) + n2
        total += n2
    return {"per_param": per_param,
            "per_group_norm": {k: v ** 0.5 for k, v in sorted(groups.items())},
            "global_norm": total ** 0.5}


def test_losses_and_gradients_match_golden(workdir):
    need("losses")
    _, _, batches, _ = decode_batches(workdir, 10, True)
    trainer = fresh_r3(workdir)
    rows = []
    for j in range(N_BATCHES):
        torch.manual_seed(SEED_LOSS + j)
        trainer.optimizer.zero_grad(set_to_none=True)
        rec = trainer.forward_backward(batches[j], unroll_steps=20)
        rows.append({"batch": j, "unroll_steps": 20, "record": scalar_record(rec),
                     "grad": grad_groups(trainer.model)})
    check_json({
        "config": "R3 preset, fresh model seed %d, history 60 s, unroll 10 s" % SEED_MODEL,
        "loss_terms": ["position", "speed", "lane", "residual_mag", "lane_rate",
                       "asym_speed", "decel_weight", "propagation"],
        "rows": rows}, "losses_r3.json")


# Parameters whose TRUE gradient is exactly zero: the last bias of the pairwise attention-bias MLP
# adds one constant per head to every logit of a softmax row, a null direction.  Autograd returns
# ~1e-9 rounding noise for it and Adam (update ~ g / (|g| + 1e-8)) turns that noise into a
# parameter step, so after three steps it differs by op-order noise (1.4e-5 observed) while every
# trained parameter agrees to 1e-5.  They are checked at NULL_DIRECTION_ATOL instead.
NULL_DIRECTION_SUFFIX = "attn/bias/2/bias"
NULL_DIRECTION_ATOL = 1e-4


def split_null_directions(arrays: dict) -> tuple[dict, dict]:
    null = {k: v for k, v in arrays.items() if k.endswith(NULL_DIRECTION_SUFFIX)}
    return {k: v for k, v in arrays.items() if k not in null}, null


def check_train3_params(actual: dict):
    expected = gc.load_npz("train3_final_params.npz")
    trained, null = split_null_directions(actual)
    trained_expected, null_expected = split_null_directions(expected)
    assert null and set(null) == set(null_expected)
    errs = gc.compare_arrays(trained, trained_expected, gc.TOLERANCES["train3_final_params.npz"],
                             "train3_final_params.npz")
    errs += gc.compare_arrays(null, null_expected, NULL_DIRECTION_ATOL,
                              "train3_final_params.npz[null-direction]")
    assert not errs, "\n".join(errs[:20])


def check_train3_json(actual: dict):
    """``train3_r3.json`` with the null-direction tensors of ``final`` moved to their own tolerance."""
    def peel(doc):
        doc = json.loads(json.dumps(doc))
        peeled = {}
        for field in ("per_tensor_sum", "per_tensor_abs_sum"):
            block = doc["final"][field]
            for key in [k for k in block if k.replace(".", "/").endswith(NULL_DIRECTION_SUFFIX)]:
                peeled[f"{field}.{key}"] = block.pop(key)
        return doc, peeled

    actual, peeled_actual = peel(write_ready(actual))
    expected, peeled_expected = peel(gc.load_json("train3_r3.json"))
    errs: list[str] = []
    gc._cmp(actual, expected, gc.TOLERANCES["train3_r3.json"], "train3_r3.json", errs,
            STRICT_HASHES)
    assert set(peeled_actual) == set(peeled_expected) and peeled_actual
    errs += [f"{k}: {peeled_actual[k]} vs {peeled_expected[k]}" for k in peeled_expected
             if abs(peeled_actual[k] - peeled_expected[k]) > NULL_DIRECTION_ATOL]
    assert not errs, "\n".join(errs[:20])


def test_three_training_steps_match_golden(workdir):
    need("train")
    _, _, batches, _ = decode_batches(workdir, 10, True)
    trainer = fresh_r3(workdir)
    before = state_checksum(trainer.model)["sha256_float32"]
    steps = []
    for j in range(3):
        torch.manual_seed(SEED_TRAIN + j)
        rec = trainer.train_batch(batches[j], unroll_steps=20)
        steps.append({"step": j + 1, "batch": j, "record": scalar_record(rec),
                      "checksum": state_checksum(trainer.model)["sha256_float32"]})
    check_train3_params(state_npz(trainer.model))
    check_train3_json({"seed": SEED_MODEL, "optimizer": "Adam lr 1e-3, grad_clip 1.0 (cfg)",
                       "init_checksum": before, "steps": steps,
                       "final": state_checksum(trainer.model)})


# --------------------------------------------------------------------------- eval
HORIZONS = (10.0, 30.0, 60.0, 120.0)
DELAY_HORIZONS = (60.0, 120.0)
EVAL_HORIZON_S = 60.0


def frames_to_arrays(frames: list[dict], prefix: str) -> dict:
    """Compact, exact array form of predicted frames (id-sorted vehicle tables)."""
    arrays = {f"{prefix}_time_s": np.asarray([float(f["time_s"]) for f in frames])}
    keys = None
    for idx, f in enumerate(frames):
        rows = sorted(f.get("vehicles", []), key=lambda r: str(r["id"]))
        if keys is None:
            keys = (sorted(k for k in rows[0]
                           if k != "id" and isinstance(rows[0][k], (int, float, bool)))
                    if rows else [])
        arrays[f"{prefix}_f{idx:04d}_ids"] = np.asarray([str(r["id"]) for r in rows])
        for k in keys:
            arrays[f"{prefix}_f{idx:04d}_{k}"] = np.asarray([float(r.get(k, np.nan)) for r in rows])
    return arrays


def test_eval_rollouts_match_golden():
    need("eval")
    from wms.eval.metrics import car_metrics
    from wms.eval.normalize import recorded_for
    from wms.eval.predictor import WmsPredictor
    from wms.eval.score import car_compact, section_delays
    from wms.eval.subsets import first_states

    receipt = json.loads((C1_DIR / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    predictor = WmsPredictor.from_checkpoint(C1_DIR / "wm_s.pt",
                                             physics=receipt["physics"]).predict_states
    results, arrays = [], {}
    for n, record in enumerate(first_states(_PATHS["eval_states_wall_same"], 3)):
        t0_s = float(record["t0_s"])
        v0 = float(record.get("v0_mps", 13.89))
        controllers = record.get("cav_controllers", {})
        arms = {}
        for arm in sorted((record.get("recorded") or {}).keys()):
            predicted = predictor(record, {**controllers, "arm": arm}, EVAL_HORIZON_S, 0)
            recorded = recorded_for(record, arm)
            car = car_compact(car_metrics(predicted, recorded, t0_s=t0_s, horizons=tuple(HORIZONS)))
            delays = {"predicted": section_delays(predicted, t0_s, v0, DELAY_HORIZONS),
                      "recorded": section_delays(recorded, t0_s, v0, DELAY_HORIZONS)}
            arms[arm] = {"car": car, "delay": delays, "n_frames": len(predicted)}
            arrays.update(frames_to_arrays(predicted, f"s{n}_{arm}"))
        results.append({"world": str(record.get("world")), "t0_s": t0_s,
                        "key": record.get("key"), "arms": arms})
    check_json({
        "states_file": "wallc1_states_v2/wall_same.json (sorted by (world,t0_s,key), first 3)",
        "horizon_s": EVAL_HORIZON_S, "cav_mode": "teacher-forced recorded arm",
        "entries": "off (checkpoint config)", "model": "C1 full_struct wm_s.pt",
        "states": results}, "eval_rollouts.json")
    check_npz(arrays, "eval_pred_frames.npz")
