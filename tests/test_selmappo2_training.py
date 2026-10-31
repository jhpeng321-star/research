"""Synthetic supervisor checks. These tests never import or start SUMO."""
import hashlib
import importlib
import json
import multiprocessing as mp
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.selmappo2 import training as driver


CELLS = tuple((p, d) for p in (.05, .1, .2, .4) for d in (1900, 2400, 2700))
FLEET_ROOT = Path("E:/agent-work/c4-selmappo/tools/selmappo")


@pytest.fixture
def shared():
    """Exercise the real loader and latest peer modules, without simulator setup."""
    if not (FLEET_ROOT / "environment.py").is_file():
        pytest.skip("fleet owner's explicit shared path is not present")
    return driver.load_shared(FLEET_ROOT.parents[1])


@pytest.fixture(autouse=True)
def synthetic_host(monkeypatch):
    monkeypatch.setattr(driver, "host_resources", lambda *args: SimpleNamespace(
        cpus=64, load=0., memory=8 * 1024**3, disk=4 * 1024**3))


@pytest.fixture
def ledger_shared(shared):
    """Synthetic per-CAV allocation; do not change the live fleet protocol."""
    protocol = SimpleNamespace(**vars(shared.protocol))
    protocol.BLOCKS = {"per_cav": (18100000, 18107999)}

    def guard(seed, variant, *, evaluation=False, smoke=False, panel=()):
        assert variant == "per_cav"
        allowed = (panel if evaluation else driver.SMOKE_SEEDS if smoke
                   else range(driver.FIRST_SEED, driver.LAST_SEED + 1))
        if type(seed) is not int or seed not in allowed:
            raise ValueError("synthetic seed allocation rejected")
        return seed

    protocol.guard_seed = guard
    protocol.guard_world = lambda world, variant, **flags: guard(world["seed"], variant, **flags)

    def training_world(seed, variant):
        guard(seed, variant)
        # Exercise the owner's real prospective cell scheduler through its
        # admitted fleet block, while preserving this synthetic per-CAV seed.
        result = shared.protocol.training_world(seed - 100000, "fleet")
        return dict(result, seed=seed)

    protocol.training_world = training_world
    return SimpleNamespace(protocol=protocol, storage=shared.storage, integration=shared.integration)


def world(index=0):
    return dict(seed=driver.FIRST_SEED + index, penetration=.2, demand=2400)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_worlds_cover_exact_fresh_allocation_and_balanced_cells(shared):
    all_worlds = []
    global_rng = driver.random.getstate()
    for iteration in range(driver.ITERATIONS):
        batch = driver.worlds(iteration, shared.protocol)
        assert batch == driver.worlds(iteration, shared.protocol)
        counts = Counter((w["penetration"], w["demand"]) for w in batch)
        assert len(batch) == 46
        assert set(counts) == set(CELLS)
        assert set(counts.values()) == {3, 4}
        all_worlds.extend(batch)
    assert driver.random.getstate() == global_rng
    seeds = [w["seed"] for w in all_worlds]
    assert seeds == list(range(18100000, 18107774))
    assert len(set(seeds)) == 169 * 46
    assert set(seeds).isdisjoint(driver.SMOKE_SEEDS)
    for bad in (-1, 169, True):
        with pytest.raises(ValueError):
            driver.worlds(bad, shared.protocol)


def test_source_receipt_covers_all_shared_python_and_local_adapters(shared):
    receipt = driver.source_receipt(shared)
    assert {path.stem for path in FLEET_ROOT.glob("*.py")} <= receipt.keys()
    assert {"guards", "dispatch", "__init__", "selmappo2.shared", "selmappo2.mixed",
            "selmappo2.training"} <= receipt.keys()
    assert all(len(digest) == 64 for digest in receipt.values())
    driver.assert_sources(shared, receipt)


class ClockContract:
    """Protocol test double: an intentionally configurable clock receipt."""
    cadence = 20
    terminal = 30
    accept_terminal = False

    def __init__(self, backend, variant):
        assert variant == "per_cav"
        self.backend, self.index = backend, 0
        self.observation = SimpleNamespace(keys=("probe-a", "probe-b"))

    def step(self, actions):
        if self.index == 30 and not self.accept_terminal:
            raise ValueError("terminal")
        self.backend.now += self.cadence
        self.backend.selected.update(tuple(driver.REQUIRED_ACTIONS)[a] for a in actions.values())
        self.index += 1
        return self.observation, -float(self.cadence), self.index == self.terminal, {}


def contract_shared(env=ClockContract, library=None):
    return SimpleNamespace(
        protocol=SimpleNamespace(LIBRARY=library or tuple(driver.REQUIRED_ACTIONS),
                                 ITERATIONS=169, EPISODES=46, guard_seed=lambda *a, **k: None),
        environment=SimpleNamespace(SelectorEnv=env))


@pytest.mark.parametrize("cadence,terminal,accept_terminal", [(60, 10, False), (20, 10, False),
                                                          (20, 30, True)])
def test_preflight_rejects_clock_terminal_and_post_terminal_drift(cadence, terminal, accept_terminal):
    env = type("Drifted", (ClockContract,), dict(cadence=cadence, terminal=terminal,
                                               accept_terminal=accept_terminal))
    with pytest.raises(ValueError, match="windows|post-terminal"):
        driver.validate_environment(contract_shared(env))


def test_preflight_requires_wall_and_all_thirty_decisions():
    with pytest.raises(ValueError, match="six actions"):
        driver.validate_environment(contract_shared(library=("native", "ctrl0", "mappo", "pi", "circles")))
    frames = driver.validate_environment(contract_shared())
    assert len(frames) == 30
    assert sum(t[2] for t in frames) == -600


def test_real_shared_environment_fails_closed_until_contract_ready(shared):
    if frozenset(shared.protocol.LIBRARY) != driver.REQUIRED_ACTIONS:
        with pytest.raises(ValueError, match="six actions"):
            driver.validate_environment(shared)
    else:
        # A peer update can expose six actions before changing the clock.
        try:
            frames = driver.validate_environment(shared)
        except ValueError as error:
            assert "windows" in str(error) or "cadence" in str(error)
        else:
            assert len(frames) == 30


def test_real_shared_thirty_frame_learning_when_contract_ready(shared):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    try:
        frames = driver.validate_environment(shared)
    except ValueError as error:
        pytest.skip(f"shared thirty-window contract pending: {error}")
    package = shared.environment.__package__
    shared.learning = importlib.import_module(f"{package}.learning")
    before = torch.get_rng_state().clone()
    driver.validate_learning(shared, frames)
    assert torch.equal(before, torch.get_rng_state())


def test_missing_shared_per_cav_allocation_is_a_clean_blocker(shared):
    if shared.protocol.BLOCKS.get("per_cav") != (18100000, 18107999):
        with pytest.raises(ValueError, match="has not admitted"):
            driver.validate_allocation(shared)
    else:
        driver.validate_allocation(shared)


def test_concurrent_reservations_use_shared_cap_and_durable_actuals(tmp_path, ledger_shared):
    budget = driver.Coordinator(tmp_path, ledger_shared)
    reservations = [budget.reserve(world(i), "training") for i in range(12)]
    assert budget.charged == 12 * 900
    assert [row["event"] for row in rows(budget.path)] == ["START"] * 12
    assert not any(row["complete"] for row in budget.records.values())
    # Runtime can close workers out of order without releasing their charges.
    for reservation in reversed(reservations):
        budget.actual(reservation["seed"], 900.)
        budget.complete(reservation["seed"])
    restored = driver.Coordinator(tmp_path, ledger_shared)
    assert restored.charged == restored.actual_sim_s == 10800
    with pytest.raises(ValueError, match="duplicate"):
        restored.reserve(world(), "training")
    assert len(rows(budget.path)) == 36


def test_failed_worker_keeps_full_reservation_and_partial_actual(tmp_path, ledger_shared):
    budget = driver.Coordinator(tmp_path, ledger_shared)
    reservation = budget.reserve(world(), "training")
    messages = []
    capability = driver.WorkerBudget(reservation, messages.append)
    capability.reserve(world(), "per_cav")
    capability.finish(317.5)
    assert messages == [dict(event="actual", seed=world()["seed"], actual_sim_s=317.5)]
    budget.actual(world()["seed"], messages[0]["actual_sim_s"])
    with pytest.raises(RuntimeError, match="dispatch stopped"):
        budget.reserve(world(1), "training")
    budget.fail(world()["seed"], "synthetic simulation fault")
    assert budget.charged == 900
    assert budget.actual_sim_s == 317.5
    assert rows(budget.path)[-1]["actual_sim_s"] == 317.5
    with pytest.raises(RuntimeError, match="dispatch stopped"):
        budget.reserve(world(1), "training")
    with pytest.raises(ValueError, match="no automatic retry"):
        driver.Coordinator(tmp_path, ledger_shared)
    with pytest.raises(ValueError, match="accounting"):
        capability.finish(318.)


def test_unresolved_actual_and_unknown_actual_block_reopening(tmp_path, ledger_shared):
    budget = driver.Coordinator(tmp_path, ledger_shared)
    budget.reserve(world(), "training")
    with pytest.raises(ValueError, match="unclosed"):
        driver.Coordinator(tmp_path, ledger_shared)
    budget.fail(world()["seed"], "worker exited before accounting")
    assert rows(budget.path)[-1]["actual_sim_s"] is None
    assert budget.charged == 900
    assert budget.actual_sim_s == 0


def test_worker_capability_is_exact_one_use_and_ledger_is_parent_only(tmp_path, ledger_shared):
    budget = driver.Coordinator(tmp_path, ledger_shared)
    reservation = budget.reserve(world(), "training")
    token = driver.WorkerBudget(reservation, lambda event: None)
    with pytest.raises(ValueError, match="capability"):
        token.reserve(world(1), "per_cav")
    with pytest.raises(ValueError, match="capability"):
        token.reserve(world(), "per_cav", evaluation=True)
    token.reserve(world(), "per_cav")
    with pytest.raises(ValueError, match="capability"):
        token.reserve(world(), "per_cav")
    with pytest.raises(ValueError, match="accounting"):
        token.finish(900.5)
    budget.owner_pid = -1
    with pytest.raises(RuntimeError, match="only the parent"):
        budget.actual(world()["seed"], 900)
    assert len(rows(budget.path)) == 1


def test_incomplete_success_and_cap_exhaustion_fail_before_dispatch(tmp_path, ledger_shared):
    budget = driver.Coordinator(tmp_path, ledger_shared)
    budget.reserve(world(), "training")
    budget.actual(world()["seed"], 899)
    with pytest.raises(ValueError, match="exactly 900"):
        budget.complete(world()["seed"])
    cap_root = tmp_path / "cap"
    cap_root.mkdir()
    capped_protocol = SimpleNamespace(**vars(ledger_shared.protocol))
    capped_protocol.CAP = 900
    capped_shared = SimpleNamespace(protocol=capped_protocol, storage=ledger_shared.storage)
    capped = driver.Coordinator(cap_root, capped_shared)
    capped.reserve(world(), "training")
    with pytest.raises(ValueError, match="exhaustion"):
        capped.reserve(world(1), "training")
    assert capped.charged == 900
    assert len(rows(capped.path)) == 1
    assert len(rows(budget.path)) == 2


def fake_result(reservation):
    training = reservation["mode"] == "training"
    return dict(event="result", seed=reservation["seed"],
                frames=[dict(reward=-20.) for _ in range(30)] if training else [],
                result=dict(world=reservation["world"], mode=reservation["mode"], decisions=30,
                            actual_sim_s=900., J600=600., episode_return=-600.))


class SyntheticProcess:
    """A thread and real pipes exercise the dispatcher without spawning SUMO."""
    def __init__(self, child, fail, reservations, barrier=None, ready=False):
        self.death_reader, self.death_writer = mp.Pipe(duplex=False)
        self.sentinel = self.death_reader

        def run():
            try:
                if ready:
                    child.send(dict(event="ready"))
                while True:
                    job = child.recv()
                    if job is None:
                        return
                    reservation = job["reservation"]
                    reservations.append(reservation)
                    if barrier is not None:
                        barrier.wait(timeout=10)
                    actual = 333. if fail else 900.
                    child.send(dict(event="actual", seed=reservation["seed"], actual_sim_s=actual))
                    if fail:
                        child.send(dict(event="error", seed=reservation["seed"], error="synthetic fault"))
                        return
                    child.send(fake_result(reservation))
            finally:
                child.close()
                self.death_writer.close()

        self.thread = threading.Thread(target=run)
        self.thread.start()

    def start(self):
        pass  # The test thread is already started; no process or simulator exists.

    def is_alive(self):
        return self.thread.is_alive()

    def join(self):
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "supervisor failed to shut down synthetic worker"
        self.death_reader.close()


def synthetic_pool(count, reservations, fail_first=False, barrier=None):
    pool = driver.Workers.__new__(driver.Workers)
    pool.workers = []
    for index in range(count):
        parent, child = mp.Pipe()
        process = SyntheticProcess(child, fail_first and index == 0, reservations, barrier)
        pool.workers.append(driver.Worker(process, parent))
    return pool


def batch(pool, candidates, budget, shared, tmp_path):
    return pool.batch(candidates, "training", {}, budget, shared, {},
                      time.monotonic() + 15, tmp_path / "STOP_REQUESTED")


def test_dispatcher_failure_stops_new_work_and_drains_inflight_actuals(tmp_path, ledger_shared, monkeypatch):
    monkeypatch.setattr(driver, "assert_sources", lambda *args: None)
    budget = driver.Coordinator(tmp_path, ledger_shared)
    reservations = []
    pool = synthetic_pool(3, reservations, fail_first=True, barrier=threading.Barrier(3))
    try:
        with pytest.raises(RuntimeError, match="synthetic fault|before 900 seconds"):
            batch(pool, [world(i) for i in range(8)], budget, ledger_shared, tmp_path)
    finally:
        pool.close(budget)
    assert len(reservations) == 3
    assert budget.charged == 2700
    assert budget.actual_sim_s == 333 + 1800
    assert all(record["complete"] for record in budget.records.values())
    assert len([row for row in rows(budget.path) if row["event"] == "START"]) == 3


def test_dispatcher_preserves_world_order_and_all_accounting(tmp_path, ledger_shared, monkeypatch):
    monkeypatch.setattr(driver, "assert_sources", lambda *args: None)
    budget = driver.Coordinator(tmp_path, ledger_shared)
    reservations = []
    pool = synthetic_pool(3, reservations)
    candidates = [world(i) for i in range(11)]
    try:
        output = batch(pool, candidates, budget, ledger_shared, tmp_path)
    finally:
        pool.close(budget)
    assert [row["seed"] for row in output] == [w["seed"] for w in candidates]
    assert len(reservations) == 11
    assert budget.actual_sim_s == budget.charged == 9900
    assert not budget.failed
    assert len(driver.Coordinator(tmp_path, ledger_shared).records) == 11


def test_stop_request_authorizes_no_new_reservations(tmp_path, ledger_shared, monkeypatch):
    monkeypatch.setattr(driver, "assert_sources", lambda *args: None)
    (tmp_path / "STOP_REQUESTED").touch()
    budget = driver.Coordinator(tmp_path, ledger_shared)
    pool = synthetic_pool(1, [])
    try:
        with pytest.raises(RuntimeError, match="cooperative stop"):
            batch(pool, [world()], budget, ledger_shared, tmp_path)
    finally:
        pool.close(budget)
    assert budget.charged == 0
    assert not budget.path.exists()


def test_shutdown_preserves_late_actual_receipts_without_retry(tmp_path, ledger_shared):
    budget = driver.Coordinator(tmp_path, ledger_shared)
    pool = synthetic_pool(1, [])
    worker = pool.workers[0]
    reservation = budget.reserve(world(), "training")
    worker.seed = world()["seed"]
    worker.connection.send(dict(reservation=reservation))
    pool.close(budget)
    pool.close(budget)
    assert budget.charged == budget.actual_sim_s == 900
    assert budget.failed
    assert rows(budget.path)[-1]["event"] == "FAILED"
    with pytest.raises(ValueError, match="no automatic retry"):
        driver.Coordinator(tmp_path, ledger_shared)


@pytest.mark.parametrize("ignore_mixed", [False, True])
def test_worker_shared_environment_inference_transport_and_learning_without_sumo(tmp_path, shared,
                                                                               ledger_shared, ignore_mixed):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    try:
        driver.validate_environment(shared)
    except ValueError as error:
        pytest.skip(f"shared thirty-window contract pending: {error}")
    learning = importlib.import_module(f"{shared.environment.__package__}.learning")
    mixed_seen = []

    class SensorBackend(driver._ProbeBackend):
        def __init__(self, *, world, variant, budget, evaluation, smoke, panel, mixed_factory, **unused):
            super().__init__()
            assert issubclass(mixed_factory, shared.runtime.MixedFleet)
            assert mixed_factory is not shared.runtime.MixedFleet
            self.env = SimpleNamespace(k=2)
            self.mixed = (shared.runtime.MixedFleet if ignore_mixed else mixed_factory)(self)
            if not ignore_mixed:
                assert type(self.mixed.handover) is not shared.runtime.PerCavHandover
            mixed_seen.append(self.mixed)
            self.actual, self.budget = 300., budget
            budget.reserve(world, variant, evaluation=evaluation, smoke=smoke, panel=panel)
            torch.manual_seed(777)  # Emulate runtime's independent controller RNG.

        def tick(self):
            self.actual += 1
            return super().tick()

        def close(self):
            self.budget.finish(self.actual)

    runtime = SimpleNamespace(**vars(shared.runtime))
    runtime.Backend = SensorBackend
    worker_shared = SimpleNamespace(**vars(shared))
    worker_shared.learning, worker_shared.runtime = learning, runtime
    model = learning.Selector().cpu()
    updater = learning.Updater(model)
    integration = tmp_path / "integration.json"
    driver.exclusive_json(integration, {})
    bindings = dict(integration_sha256=shared.protocol.digest(integration))
    path = driver.save_checkpoint(tmp_path, 0, model, updater, {}, bindings)
    digest = shared.protocol.digest(path)
    budget = driver.Coordinator(tmp_path, ledger_shared)
    reservation = budget.reserve(world(), "training")
    messages = []
    job = dict(
        reservation=reservation, checkpoint=str(path), checkpoint_sha256=digest, sources={},
        iteration=0, out=str(tmp_path), registration="unused", c1_root="unused",
        integration=str(integration), design_sha="unused", bindings=bindings)
    if ignore_mixed:
        with pytest.raises(ValueError, match="required per-CAV mixed fleet"):
            driver.run_episode(worker_shared, job, messages.append)
        assert messages == [dict(event="actual", seed=world()["seed"], actual_sim_s=300.)]
        assert not mixed_seen[0].backend.names  # no selection before rejecting the base fleet
        budget.actual(world()["seed"], 300.)
        budget.fail(world()["seed"], "missing mixed fleet")
        assert budget.charged == 900 and budget.failed
        return
    output = driver.run_episode(worker_shared, job, messages.append)
    assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(777).get_state())
    assert messages == [dict(event="actual", seed=world()["seed"], actual_sim_s=900.)]
    event = dict(event="result", seed=world()["seed"], **output)
    driver.validate_result(event, reservation)
    budget.actual(world()["seed"], 900.)
    budget.complete(world()["seed"])
    assert output["result"]["J600"] == 600
    assert output["result"]["episode_return"] == -600
    assert len(mixed_seen) == 1
    assert type(mixed_seen[0]).__module__ == "tools.selmappo2.mixed"
    trajectory = driver.decode_trajectory(worker_shared, output["frames"])
    before = [p.detach().clone() for p in model.parameters()]
    stats = updater.update([trajectory])
    assert stats["frames"] == 30
    assert stats["samples"] == 60
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert shared.protocol.digest(path) == digest


def test_checkpoint_is_immutable_and_retains_optimization_state(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = torch.nn.Linear(2, 1)
    updater = SimpleNamespace(actor_opt=torch.optim.Adam(model.parameters()),
                              critic_opt=torch.optim.Adam(model.parameters()),
                              scaler=SimpleNamespace(count=7, mean=2., std=3.))
    path = driver.save_checkpoint(tmp_path, 0, model, updater, {"learning": "frozen-sha"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        driver.save_checkpoint(tmp_path, 0, model, updater, {})
    assert path.read_bytes() == original
    driver.save_checkpoint(tmp_path, 1, model, updater, {"learning": "frozen-sha"})
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["iteration"] == 0
    assert saved["optimization_seed"] == 1
    assert saved["variant"] == "per_cav"
    assert saved["scaler"]["count"] == 7
    assert saved["actor_opt"] and saved["critic_opt"]
    assert {"numpy_rng", "torch_rng", "python_rng"} <= saved.keys()
    assert len(list(tmp_path.glob("checkpoint_*.pt"))) == 2


def continuation_fixture():
    from tools.selmappo2 import ops
    seed = ops.HISTORICAL_SEED
    events = [dict(event="START", seed=seed, mode="smoke",
                   world=dict(seed=seed, penetration=.4, demand=2700), reserved_sim_s=900, panel=[]),
              dict(event="ACTUAL", seed=seed, actual_sim_s=300.),
              dict(event="FAILED", seed=seed, actual_sim_s=300., error=ops.HISTORICAL_ERROR)]
    raw = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in events).encode()
    assert ops.sha(raw) == ops.HISTORICAL_LEDGER_SHA
    receipt = dict(original=dict(ledger_sha256=ops.HISTORICAL_LEDGER_SHA),
                   inherited_reserved_sim_s=900, inherited_actual_sim_s=300,
                   skipped_seed=seed, only_unused_smoke_seed=18107999)
    return raw, receipt


def test_acknowledged_history_keeps_charge_skips_failed_seed_and_requires_second_smoke(tmp_path, ledger_shared):
    raw, receipt = continuation_fixture()
    (tmp_path / "coordinator.jsonl").write_bytes(raw)
    budget = driver.Coordinator(tmp_path, ledger_shared, continuation=receipt)
    assert budget.charged == 900 and budget.actual_sim_s == 300 and not budget.failed
    with pytest.raises(ValueError, match="duplicate"):
        budget.reserve(dict(seed=18107998, penetration=.4, demand=2700), "smoke")
    pool = driver.Workers.__new__(driver.Workers)
    pool.workers, pool.limit = [], 1
    pool._start_worker = lambda: pool.workers.append(SimpleNamespace(seed=None))
    with pytest.raises(RuntimeError, match="smoke"):
        pool.expand(budget)
    budget.reserve(dict(seed=18107999, penetration=.4, demand=2700), "smoke")
    budget.actual(18107999, 900.)
    budget.complete(18107999)
    pool.expand(budget)
    assert len(pool.workers) == 1
    assert budget.charged == 1800 and budget.actual_sim_s == 1200
    assert budget.path.read_bytes().startswith(raw)


def test_continuation_does_not_exempt_another_failure_or_an_incomplete_reservation(tmp_path, ledger_shared):
    raw, receipt = continuation_fixture()
    (tmp_path / "coordinator.jsonl").write_bytes(raw)
    budget = driver.Coordinator(tmp_path, ledger_shared, continuation=receipt)
    budget.reserve(dict(seed=18107999, penetration=.4, demand=2700), "smoke")
    budget.actual(18107999, 899.)
    budget.fail(18107999, "another fault")
    with pytest.raises(ValueError, match="no automatic retry"):
        driver.Coordinator(tmp_path, ledger_shared, continuation=receipt)
    with pytest.raises(RuntimeError, match="dispatch stopped"):
        budget.reserve(world(), "training")
    assert budget.charged == 1800 and budget.actual_sim_s == 1199


def test_continuation_rejects_tampered_historical_bytes_and_inherited_charge(tmp_path, ledger_shared):
    raw, receipt = continuation_fixture()
    path = tmp_path / "coordinator.jsonl"
    path.write_bytes(raw.replace(b"300.0", b"301.0"))
    with pytest.raises(ValueError, match="historical ledger"):
        driver.Coordinator(tmp_path, ledger_shared, continuation=receipt)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="historical ledger"):
        driver.Coordinator(tmp_path, ledger_shared, continuation=dict(receipt, inherited_reserved_sim_s=0))
    with pytest.raises(ValueError, match="no automatic retry"):
        driver.Coordinator(tmp_path, ledger_shared)


def test_continuation_authority_is_specific_and_one_use_lock_blocks_before_spawn(tmp_path):
    from tools.selmappo2 import ops
    reg = tmp_path / "REGISTRATION.md"
    reg.write_text("- 01:48 MAIN: fleet continuation acknowledged\n")
    with pytest.raises(PermissionError):
        ops.continuation_authority(reg)
    authority = ("- 01:48 MAIN ACKNOWLEDGEMENTS / DECISIONS: (2) S-SELMAPPO2 same-cap "
                 "continuation ACKNOWLEDGED (unused second smoke world only, then production).")
    reg.write_text(authority + "\n")
    assert ops.continuation_authority(reg) == authority
    lock = tmp_path / ops.CONTINUATION_LOCK
    ops.exclusive_json(lock, dict(consumed=True))
    with pytest.raises(FileExistsError, match="one-use continuation lock consumed"):
        ops._remote_dispatch(tmp_path / ("build_" + "a" * 64), {}, "a" * 64, continuation=True)
    assert ops.read_json(lock) == dict(consumed=True)


def test_per_cav_worker_wire_contains_no_tensor_storage_descriptors():
    """Actual multiprocessing Pipe serialization stays ordinary inline data."""
    from multiprocessing.reduction import ForkingPickler
    torch = pytest.importorskip("torch")
    shared_obs = SimpleNamespace(keys=("a", "b"), actors=((0., 1.), (1., 0.)), shared=(0.,))
    frame = dict(keys=shared_obs.keys, actors=shared_obs.actors, shared=shared_obs.shared,
                 actions=dict(a=0, b=5), old_logp=torch.zeros(2).tolist(), reward=-20.)
    trajectories = [dict(frames=[frame] * 30, result=dict(J600=600.)) for _ in range(46)]
    raw = bytes(ForkingPickler.dumps(trajectories))
    assert b"rebuild_storage" not in raw and b"rebuild_tensor" not in raw
    assert ForkingPickler.loads(raw) == trajectories


def test_evaluation_rejects_non_final_checkpoint_before_runtime(tmp_path, shared):
    torch = pytest.importorskip("torch")
    path = tmp_path / "checkpoint_168.pt"
    torch.save(dict(iteration=168, variant="per_cav", optimization_seed=1, sources={}), path)
    panel_world = dict(seed=1, penetration=.4, demand=2700)
    reservation = dict(seed=1, world=panel_world, mode="evaluation")
    job = dict(reservation=reservation, checkpoint=str(path),
               checkpoint_sha256=shared.protocol.digest(path), iteration=168, sources={})
    with pytest.raises(ValueError, match="only frozen final169"):
        driver.run_episode(shared, job, lambda event: None)


def test_evaluation_requires_explicit_frozen_matching_panel(tmp_path, ledger_shared):
    shared = ledger_shared
    path = tmp_path / "evaluation_contract.json"
    panel = [dict(seed=19000000 + index, penetration=p, demand=d)
             for index, (p, d) in enumerate(shared.protocol.PANEL_CELLS)]
    registration = tmp_path / "synthetic_registry.txt"
    registration.write_text("- S-C1SWITCH2 DESIGN REGISTERED SHA256 " + "b" * 64 + "\n",
                            encoding="utf-8")
    driver.exclusive_json(path, dict(protocol=shared.protocol.PROTOCOL, worlds=panel,
                                     design_sha256="b" * 64))
    with pytest.raises(ValueError, match="frozen"):
        driver.evaluation_worlds(shared, path, None, registration)
    with pytest.raises(ValueError, match="hash mismatch"):
        driver.evaluation_worlds(shared, path, "0" * 64, registration)
    actual, digest = driver.evaluation_worlds(shared, path, shared.protocol.digest(path), registration)
    assert actual == panel
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    mismatch = tmp_path / "mismatch_contract.json"
    changed = [dict(w) for w in panel]
    changed[0]["demand"] = 3000
    driver.exclusive_json(mismatch, dict(protocol=shared.protocol.PROTOCOL, worlds=changed,
                                         design_sha256="b" * 64))
    with pytest.raises(ValueError, match="disagree"):
        driver.evaluation_worlds(shared, mismatch, shared.protocol.digest(mismatch), registration)
    unregistered = tmp_path / "unregistered_registry.txt"
    unregistered.write_text("synthetic input without peer design\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="not registered"):
        driver.evaluation_worlds(shared, path, shared.protocol.digest(path), unregistered)


def test_result_validation_and_team_return_statistics():
    reservation = dict(seed=world()["seed"], world=world(), mode="training")
    output = fake_result(reservation)
    driver.validate_result(output, reservation)
    stats = driver.return_statistics([output])
    assert stats["J600_mean"] == 600
    assert stats["episode_return_mean"] == -600
    assert stats["episode_return_std"] == 0
    output["frames"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        driver.validate_result(output, reservation)


def test_worker_limit_checked_before_starting_processes():
    for count in (0, 13, True):
        with pytest.raises(ValueError, match="workers"):
            driver.Workers(count, "unused", {})


def synthetic_context(created, reservations, fail=False):
    class Context:
        @staticmethod
        def Pipe():
            parent, child = mp.Pipe()
            # Spawn transfers a separate child handle; a test thread shares its
            # raw handle, so only the thread closes it here.
            return parent, SimpleNamespace(raw=child, close=lambda: None)

        @staticmethod
        def Process(*, target, args):
            assert target is driver._worker_main
            process = SyntheticProcess(args[0].raw, fail, reservations, ready=True)
            created.append(process)
            return process

    return Context()


def test_both_smokes_run_on_one_worker_before_expanding_to_twelve(tmp_path, ledger_shared, monkeypatch):
    created, reservations = [], []
    context = synthetic_context(created, reservations)
    monkeypatch.setattr(driver.mp, "get_context", lambda method: context if method == "spawn" else None)
    monkeypatch.setattr(driver, "assert_sources", lambda *args: None)
    budget = driver.Coordinator(tmp_path, ledger_shared)
    pool = driver.Workers(12, "unused", {})
    try:
        assert len(created) == len(pool.workers) == 1
        with pytest.raises(RuntimeError, match="single-worker"):
            pool.expand(budget)
        for index, seed in enumerate(driver.SMOKE_SEEDS):
            pool.batch([dict(seed=seed, penetration=.4, demand=2700)], "smoke", {}, budget,
                       ledger_shared, {}, time.monotonic() + 15, tmp_path / "STOP_REQUESTED")
            assert len(created) == 1
            if index == 0:
                with pytest.raises(RuntimeError, match="single-worker"):
                    pool.expand(budget)
        pool.expand(budget)
        assert len(created) == len(pool.workers) == 12
        pool.expand(budget)
        assert len(created) == 12
        assert [r["seed"] for r in reservations] == list(driver.SMOKE_SEEDS)
        assert budget.charged == budget.actual_sim_s == 1800
    finally:
        pool.close(budget)


def test_failed_first_canary_never_expands_or_dispatches_second(tmp_path, ledger_shared, monkeypatch):
    created, reservations = [], []
    context = synthetic_context(created, reservations, fail=True)
    monkeypatch.setattr(driver.mp, "get_context", lambda method: context)
    monkeypatch.setattr(driver, "assert_sources", lambda *args: None)
    budget = driver.Coordinator(tmp_path, ledger_shared)
    pool = driver.Workers(12, "unused", {})
    try:
        with pytest.raises(RuntimeError, match="synthetic fault|before 900"):
            pool.batch([dict(seed=seed, penetration=.4, demand=2700) for seed in driver.SMOKE_SEEDS],
                       "smoke", {}, budget, ledger_shared, {}, time.monotonic() + 15,
                       tmp_path / "STOP_REQUESTED")
        with pytest.raises(RuntimeError, match="single-worker"):
            pool.expand(budget)
        assert len(created) == len(reservations) == 1
        assert budget.charged == 900 and budget.actual_sim_s == 333
        assert driver.SMOKE_SEEDS[1] not in budget.records
    finally:
        pool.close(budget)


@pytest.mark.parametrize("cpus,load,pending,disk,memory,admitted", [
    (64, 23., 0, 2, 8, True),   # projected load exactly 24
    (64, 23.1, 0, 2, 8, False),
    (8, 5., 0, 2, 8, True),    # projected load exactly cpus - 2
    (8, 5., 1, 2, 8, False),   # an outstanding reservation consumes capacity
    (64, 0., 0, 1.99, 8, False),
    (64, 0., 0, 2, 7.99, False),
    (2, 0., 0, 2, 8, False),
    (None, 0., 0, 2, 8, False),
    (64, float("nan"), 0, 2, 8, False),
])
def test_resource_admission_boundaries(cpus, load, pending, disk, memory, admitted, tmp_path,
                                      shared, monkeypatch):
    monkeypatch.setattr(driver, "host_resources", lambda *args: SimpleNamespace(
        cpus=cpus, load=load, memory=memory * 1024**3, disk=disk * 1024**3))
    assert driver.resource_admitted(tmp_path, shared.protocol, pending) is admitted


def test_resource_pressure_yields_before_reserving_and_then_dispatches(tmp_path, ledger_shared, monkeypatch):
    monkeypatch.setattr(driver, "assert_sources", lambda *args: None)
    monkeypatch.setattr(driver, "RESOURCE_WAIT_SECONDS", .001)
    budget = driver.Coordinator(tmp_path, ledger_shared)
    snapshots = []

    def resources(*args):
        busy = len(snapshots) < 2
        if busy:
            assert not budget.records and budget.charged == 0
        snapshots.append(busy)
        return SimpleNamespace(cpus=64, load=30. if busy else 0.,
                               memory=8 * 1024**3, disk=4 * 1024**3)

    monkeypatch.setattr(driver, "host_resources", resources)
    reservations = []
    pool = synthetic_pool(1, reservations)
    try:
        outputs = batch(pool, [world()], budget, ledger_shared, tmp_path)
    finally:
        pool.close(budget)
    assert snapshots == [True, True, False]
    assert len(outputs) == len(reservations) == 1
    assert budget.charged == budget.actual_sim_s == 900


@pytest.mark.parametrize("stop", [False, True])
def test_resource_wait_obeys_deadline_and_cooperative_stop(tmp_path, ledger_shared, monkeypatch, stop):
    monkeypatch.setattr(driver, "assert_sources", lambda *args: None)
    monkeypatch.setattr(driver, "RESOURCE_WAIT_SECONDS", .001)
    budget = driver.Coordinator(tmp_path, ledger_shared)
    stop_path = tmp_path / "STOP_REQUESTED"

    def resources(*args):
        if stop:
            stop_path.touch()
        # A stop arriving during an otherwise successful resource probe must
        # still precede any durable reservation or send.
        return SimpleNamespace(cpus=64, load=0. if stop else 30.,
                               memory=8 * 1024**3, disk=4 * 1024**3)

    monkeypatch.setattr(driver, "host_resources", resources)
    pool = synthetic_pool(1, [])
    try:
        with pytest.raises(RuntimeError, match="wall limit or cooperative stop"):
            pool.batch([world()], "training", {}, budget, ledger_shared, {},
                       time.monotonic() + .03, stop_path)
    finally:
        pool.close(budget)
    assert budget.charged == 0 and not budget.path.exists()


def test_main_audits_and_loads_integration_for_exact_per_cav_design(shared, monkeypatch):
    calls = []
    proxy = SimpleNamespace(**vars(shared))
    proxy.protocol = SimpleNamespace(**vars(shared.protocol))
    proxy.integration = SimpleNamespace(**vars(shared.integration))
    proxy.protocol.approval_line = lambda path: "synthetic approval"

    def audit(path, *, variant, design_sha):
        calls.append(("audit", path, variant, design_sha))
        return dict(passed=True)

    class PreflightReached(Exception):
        pass

    def integration(path, registration, root, design_sha, *, variant):
        calls.append(("integration", registration, variant, design_sha))
        raise PreflightReached

    proxy.protocol.audit_registration = audit
    proxy.integration.load = integration
    monkeypatch.setattr(driver, "configure_cpu", lambda: None)
    monkeypatch.setattr(driver, "load_shared", lambda root: proxy)
    with pytest.raises(PreflightReached):
        driver.main(["--shared-root", "unused", "--registration", "synthetic-registry",
                     "--c1-root", "unused", "--integration", "unused", "--design-sha", "a" * 64,
                     "--out", "unused"])
    assert calls == [("audit", "synthetic-registry", "per_cav", "a" * 64),
                     ("integration", "synthetic-registry", "per_cav", "a" * 64)]
