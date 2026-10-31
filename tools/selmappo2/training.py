"""Per-CAV supervisor; all simulation, observations and PPO belong to selmappo.

The shared loader is supplied by tools.selmappo2.shared. No compatibility patches
are applied to the fleet owner's modules. A behavioral preflight runs without
SUMO before creating a run. Training is resumeless; failed reservations are never
refunded or retried. Evaluation is a separate, explicitly contracted operation.
"""
import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from types import SimpleNamespace


VARIANT = "per_cav"
OPTIMIZATION_SEED = 1
ITERATIONS, EPISODES, DECISIONS = 169, 46, 30
CADENCE, WARMUP, DURATION = 20, 300, 900
FIRST_SEED, LAST_SEED = 18100000, 18107773
SMOKE_SEEDS = (18107998, 18107999)
MAX_WORKERS, WALL_SECONDS = 12, 86400
CPU_HEADROOM, LOAD_CAP = 2, 24
MIN_DISK_BYTES, MIN_MEMORY_BYTES = 2 * 1024**3, 4 * 1024**3
RESOURCE_WAIT_SECONDS = 1.
REQUIRED_ACTIONS = frozenset(("native", "ctrl0", "mappo", "pi", "circles", "wall"))
SOURCE_FILES = ("environment", "observations", "learning", "protocol", "storage",
                "runtime", "integration", "handover", "guards", "dispatch", "train", "remote")


def load_shared(root):
    from .shared import load
    package = load(root)
    return SimpleNamespace(**{
        name: importlib.import_module(f"{package.__name__}.{name}")
        for name in SOURCE_FILES if name not in ("train", "remote", "handover", "dispatch")
    })


def source_receipt(shared):
    root = shared.protocol.safe_path(Path(shared.environment.__file__).parent)
    files = {path.stem: shared.protocol.safe_path(path) for path in root.glob("*.py")}
    if not {*SOURCE_FILES, "__init__"} <= files.keys():
        raise ValueError("required shared Python sources are missing")
    receipt = {name: shared.protocol.digest(path) for name, path in sorted(files.items())}
    for name in ("shared", "mixed", "training"):
        receipt[f"selmappo2.{name}"] = shared.protocol.digest(Path(__file__).with_name(f"{name}.py"))
    return receipt


def assert_sources(shared, expected):
    if source_receipt(shared) != expected:
        raise ValueError("shared sources changed during run; stop dispatch")


def validate_allocation(shared):
    if shared.protocol.BLOCKS.get(VARIANT) != (FIRST_SEED, SMOKE_SEEDS[-1]):
        raise ValueError("shared protocol has not admitted the exact per-CAV allocation")
    if shared.protocol.CAP != (ITERATIONS * EPISODES + len(SMOKE_SEEDS) + 16) * DURATION:
        raise ValueError("shared budget does not match the full per-CAV allocation")
    if not all(callable(getattr(shared.protocol, name, None))
               for name in ("training_world", "guard_world")):
        raise ValueError("shared prospective world schedule/guard is unavailable")
    for iteration in (0, ITERATIONS - 1):
        for world in worlds(iteration, shared.protocol):
            shared.protocol.guard_world(world, VARIANT)
    for seed in SMOKE_SEEDS:
        shared.protocol.guard_world(dict(seed=seed, penetration=.4, demand=2700), VARIANT, smoke=True)


def worlds(iteration, protocol):
    """Use the shared prospective schedule for each fresh per-CAV seed."""
    if type(iteration) is not int or not 0 <= iteration < ITERATIONS:
        raise ValueError("iteration outside training allocation")
    start = FIRST_SEED + iteration * EPISODES
    result = [protocol.training_world(seed, VARIANT) for seed in range(start, start + EPISODES)]
    if [w["seed"] for w in result] != list(range(start, start + EPISODES)):
        raise ValueError("shared world schedule changed the per-CAV seed allocation")
    return result


class _ProbeBackend:
    """Sensor-only clock fixture, never a simulator or an environment substitute."""
    def __init__(self):
        self.now = WARMUP
        self.names = {}
        self.selected = set()

    def observable(self):
        return {"history": [{"now_s": self.now, "self_reports": [
            dict(vehicle_id=key, timestamp_s=self.now, lane=i, s_m=100.,
                 v_mps=20., a_mps2=0.) for i, key in enumerate(("probe-a", "probe-b"))
        ], "detections": [], "neighbours": [], "coverage": []}]}

    def select_per_cav(self, names):
        self.names = dict(names)
        self.selected.update(names.values())

    def current_names(self):
        return self.names

    def tick(self):
        self.now += 1
        return dict(speed_deficit_vehicle_seconds=1., source_wait_vehicle_seconds=0.)


def validate_environment(shared):
    """Exercise the actual shared environment, including terminal rejection."""
    library = tuple(shared.protocol.LIBRARY)
    if len(library) != 6 or frozenset(library) != REQUIRED_ACTIONS:
        raise ValueError("shared contract requires six actions including wall")
    if (shared.protocol.ITERATIONS, shared.protocol.EPISODES) != (ITERATIONS, EPISODES):
        raise ValueError("shared update/episode allocation mismatch")
    backend = _ProbeBackend()
    env = shared.environment.SelectorEnv(backend, VARIANT)
    frames = []
    for decision in range(DECISIONS):
        obs = env.observation
        if tuple(obs.keys) != ("probe-a", "probe-b"):
            raise ValueError("shared environment is not per-CAV")
        actions = {key: (decision + i) % 6 for i, key in enumerate(obs.keys)}
        _, reward, done, _ = env.step(actions)
        if (backend.now != WARMUP + CADENCE * (decision + 1)
                or done != (decision == DECISIONS - 1) or reward != -float(CADENCE)):
            raise ValueError("shared environment must use thirty 20-second team-reward windows")
        frames.append((obs, actions, reward))
    if backend.now != DURATION or backend.selected != REQUIRED_ACTIONS:
        raise ValueError("shared cadence/action coverage mismatch")
    try:
        env.step({key: 0 for key in env.observation.keys})
    except ValueError:
        pass
    else:
        raise ValueError("shared environment accepted a post-terminal action")
    return frames


def validate_learning(shared, frames):
    """Reject stale five-logit/ten-frame PPO without modifying shared code."""
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(OPTIMIZATION_SEED)
        model = shared.learning.Selector().cpu()
        updater = shared.learning.Updater(model)
        transitions = []
        generator = torch.Generator().manual_seed(OPTIMIZATION_SEED)
        before = [p.detach().clone() for p in model.parameters()]
        for obs, _, reward in frames:
            actors = torch.tensor(obs.actors, dtype=torch.float32)
            if model.distribution(actors).logits.shape != (len(obs.keys), 6):
                raise ValueError("shared selector must expose six categorical logits")
            actions, logp = model.act(obs, generator=generator)
            transitions.append(shared.learning.Transition(obs, actions, logp, reward))
        stats = updater.update([transitions])
        if stats.get("frames") != DECISIONS or not all(
                math.isfinite(float(value)) for value in stats.values()):
            raise ValueError("shared learning rejected complete thirty-window trajectories")
        if not any(not torch.equal(a, b) for a, b in zip(before, model.parameters())):
            raise ValueError("shared PPO preflight did not learn")


def configure_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be empty")
    if not hasattr(os, "getpriority") or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError("CPU-only nice19 process required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def host_resources(root, protocol):
    """Linux resource receipts only; no simulator or workload inspection."""
    available = next(int(line.split()[1]) * 1024 for line in
                     protocol.safe_path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
                     if line.startswith("MemAvailable:"))
    return SimpleNamespace(cpus=os.cpu_count(), load=os.getloadavg()[0],
                           memory=available, disk=shutil.disk_usage(root).free)


def resource_admitted(root, protocol, pending):
    """Leave two cores free, cap load at 24, and include unreflected dispatches.

    Existing work already contributes to load, but counting pending episodes as
    well prevents a fast burst of reservations from outrunning the load receipt.
    Pressure or unavailable resource receipts yield until stop/deadline.
    """
    try:
        state = host_resources(root, protocol)
        return (state.cpus is not None and state.cpus > CPU_HEADROOM
                and math.isfinite(state.load) and state.load >= 0
                and state.load + pending + 1 <= min(state.cpus - CPU_HEADROOM, LOAD_CAP)
                and state.disk >= MIN_DISK_BYTES and state.memory >= MIN_MEMORY_BYTES)
    except (OSError, ValueError, StopIteration, AttributeError):
        return False


def exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint_path(root, iteration):
    if type(iteration) is not int or not 0 <= iteration <= ITERATIONS:
        raise ValueError("invalid checkpoint iteration")
    return Path(root) / f"checkpoint_{iteration:03d}.pt"


def save_checkpoint(root, iteration, model, updater, sources, bindings=None):
    import numpy as np
    import torch
    path = checkpoint_path(root, iteration)
    payload = dict(model=model.state_dict(), actor_opt=updater.actor_opt.state_dict(),
                   critic_opt=updater.critic_opt.state_dict(), scaler=vars(updater.scaler),
                   iteration=iteration, variant=VARIANT, optimization_seed=OPTIMIZATION_SEED,
                   sources=sources, bindings=bindings or {}, torch_rng=torch.get_rng_state(),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state())
    # Opening exclusively also protects a checkpoint left incomplete by a crash.
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def evaluation_worlds(shared, contract_path, contract_sha, registration):
    """A caller-pinned, shared-validated manifest is the explicit panel freeze."""
    path = shared.protocol.safe_path(contract_path)
    if not contract_sha:
        raise ValueError("evaluation requires an explicit frozen contract SHA256")
    panel = shared.integration.evaluation_panel(path, contract_sha, registration=registration)
    if tuple((w["penetration"], w["demand"]) for w in panel) != tuple(shared.protocol.PANEL_CELLS):
        raise ValueError("frozen contract and shared panel cells disagree")
    seeds = tuple(w["seed"] for w in panel)
    if any(FIRST_SEED <= seed <= SMOKE_SEEDS[-1] for seed in seeds):
        raise ValueError("evaluation intersects per-CAV allocation")
    for world in panel:
        shared.protocol.guard_world(world, VARIANT, evaluation=True, panel=seeds)
    return panel, shared.protocol.digest(path)


class Coordinator:
    """Parent-only concurrent counterpart of shared Budget's single pending slot.

    Uses shared guards, cap and fsynced append. START always precedes dispatch.
    ACTUAL comes from runtime.close via the capability; RESULT is separate because
    a runtime can close successfully while the trajectory itself has failed.
    An incomplete, failed or duplicate reservation permanently blocks resumption.
    """
    def __init__(self, root, shared, panel=()):
        self.shared = shared
        self.panel = tuple(panel)
        self.path = shared.protocol.safe_path(shared.protocol.safe_path(root) / "coordinator.jsonl")
        self.owner_pid = os.getpid()
        self.charged = 0
        self.records = {}
        self.failed = False
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line))
            if self.failed or any(not row["complete"] for row in self.records.values()):
                raise ValueError("failed/unclosed reservation; no automatic retry or resumption")

    def _apply(self, row):
        seed, event = row["seed"], row["event"]
        if event == "START":
            if seed in self.records or row["reserved_sim_s"] != DURATION:
                raise ValueError("duplicate/invalid reservation")
            self.shared.protocol.guard_world(row["world"], VARIANT,
                                             evaluation=row["mode"] == "evaluation",
                                             smoke=row["mode"] == "smoke", panel=self.panel)
            if row["world"]["seed"] != seed or row["mode"] not in ("training", "smoke", "evaluation"):
                raise ValueError("invalid reservation world/mode")
            if tuple(row["panel"]) != (self.panel if row["mode"] == "evaluation" else ()):
                raise ValueError("reservation panel does not match explicit freeze")
            if self.charged + DURATION > self.shared.protocol.CAP:
                raise ValueError("budget exhaustion")
            self.charged += DURATION
            self.records[seed] = dict(reservation=row, actual=None, complete=False)
        else:
            record = self.records[seed]
            if record["complete"]:
                raise ValueError("duplicate terminal receipt")
            if event == "ACTUAL":
                actual = row["actual_sim_s"]
                if (record["actual"] is not None or isinstance(actual, bool)
                        or not math.isfinite(actual) or not 0 <= actual <= DURATION):
                    raise ValueError("invalid actual simulation accounting")
                record["actual"] = actual
                if actual != DURATION:
                    self.failed = True
            elif event == "RESULT":
                if record["actual"] != DURATION:
                    raise ValueError("successful episode must account for exactly 900 seconds")
                record["complete"] = True
            elif event == "FAILED":
                if row["actual_sim_s"] != record["actual"]:
                    raise ValueError("failed accounting mismatch")
                self.failed = True
                record["complete"] = True
            else:
                raise ValueError("invalid coordinator event")

    def _write(self, row):
        if os.getpid() != self.owner_pid:
            raise RuntimeError("only the parent may write the budget ledger")
        # Validate before writing, then fsync before returning to the dispatcher.
        self._apply(row)
        try:
            self.shared.storage.append(self.path, row)
        except BaseException:
            self.failed = True
            raise

    def reserve(self, world, mode):
        if self.failed:
            raise RuntimeError("failed simulation: dispatch stopped")
        row = dict(event="START", seed=world["seed"], world=dict(world), mode=mode,
                   reserved_sim_s=DURATION, panel=self.panel if mode == "evaluation" else ())
        self._write(row)
        return row

    def actual(self, seed, actual):
        self._write(dict(event="ACTUAL", seed=seed, actual_sim_s=actual))

    def complete(self, seed):
        self._write(dict(event="RESULT", seed=seed))

    def fail(self, seed, error):
        self._write(dict(event="FAILED", seed=seed,
                         actual_sim_s=self.records[seed]["actual"], error=error))

    @property
    def actual_sim_s(self):
        return sum(row["actual"] for row in self.records.values() if row["actual"] is not None)


class WorkerBudget:
    """One-use capability, not a second ledger or worker-side budget allocator."""
    def __init__(self, reservation, send):
        self.reservation, self.send = reservation, send
        self.started, self.finished = False, False

    def reserve(self, world, variant, *, evaluation=False, smoke=False, panel=()):
        expected = self.reservation
        if (self.started or variant != VARIANT or world != expected["world"]
                or evaluation != (expected["mode"] == "evaluation")
                or smoke != (expected["mode"] == "smoke") or tuple(panel) != tuple(expected["panel"])):
            raise ValueError("runtime exceeded its parent reservation capability")
        self.started = True

    def finish(self, actual):
        if (not self.started or self.finished or isinstance(actual, bool)
                or not math.isfinite(actual) or not 0 <= actual <= DURATION):
            raise ValueError("invalid worker terminal accounting")
        self.send(dict(event="actual", seed=self.reservation["seed"], actual_sim_s=actual))
        self.finished = True


def run_episode(shared, job, send):
    """Worker inference on a frozen parent checkpoint; shared backend owns SUMO."""
    import torch
    from .mixed import make_mixed_fleet
    reservation = job["reservation"]
    world, mode = reservation["world"], reservation["mode"]
    path = shared.protocol.safe_path(job["checkpoint"])
    if shared.protocol.digest(path) != job["checkpoint_sha256"]:
        raise ValueError("checkpoint changed before worker inference")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["variant"] != VARIANT or saved["optimization_seed"] != OPTIMIZATION_SEED
            or saved["sources"] != job["sources"] or saved["iteration"] != job["iteration"]
            or saved.get("bindings", {}) != job.get("bindings", {})):
        raise ValueError("checkpoint contract mismatch")
    if mode == "evaluation" and saved["iteration"] != ITERATIONS:
        raise ValueError("only frozen final169 may be evaluated")
    if shared.protocol.digest(job["integration"]) != job["bindings"]["integration_sha256"]:
        raise ValueError("integration receipt changed since checkpoint freeze")
    model = shared.learning.Selector().cpu()
    model.load_state_dict(saved["model"])
    model.eval()
    # Runtime seeds its controller RNG. Keep selector sampling independent of
    # that stream and deterministic per world, independent of worker scheduling.
    policy_rng = torch.Generator(device="cpu").manual_seed(world["seed"])
    root = shared.protocol.safe_path(job["out"]) / str(world["seed"])
    root.mkdir(exist_ok=False)
    capability = WorkerBudget(reservation, send)
    backend = None
    begin = time.monotonic()
    frames, rewards, metrics = [], [], {}
    try:
        mixed_factory = make_mixed_fleet(shared.runtime)
        backend = shared.runtime.Backend(world=world, variant=VARIANT,
                                        registration=job["registration"], c1_root=job["c1_root"],
                                        episode_root=root, budget=capability,
                                        integration=job["integration"], design_sha=job["design_sha"],
                                        evaluation=mode == "evaluation", smoke=mode == "smoke",
                                        panel=tuple(reservation["panel"]),
                                        mixed_factory=mixed_factory)
        if not isinstance(backend.mixed, mixed_factory):
            raise ValueError("shared backend did not install the required per-CAV mixed fleet")
        env = shared.environment.SelectorEnv(backend, VARIANT)
        for decision in range(DECISIONS):
            if backend.now != WARMUP + decision * CADENCE:
                raise ValueError("off-cadence shared runtime")
            obs = env.observation
            with torch.no_grad():
                actions, logp = model.act(obs, deterministic=mode == "evaluation", generator=policy_rng)
            if mode == "smoke":
                actions = {key: (decision + i) % 6 for i, key in enumerate(obs.keys)}
            _, reward, done, components = env.step(actions)
            if (backend.now != WARMUP + (decision + 1) * CADENCE
                    or done != (decision == DECISIONS - 1) or not math.isfinite(reward)):
                raise ValueError("shared cadence/terminal/reward mismatch")
            rewards.append(float(reward))
            if mode == "training":
                frames.append(dict(keys=obs.keys, actors=obs.actors, shared=obs.shared,
                                   actions=actions, old_logp=logp.detach().cpu().tolist(), reward=reward))
            for key, value in components.items():
                metrics[key] = metrics.get(key, 0.) + float(value)
        if not capability.started or backend.actual != DURATION:
            raise ValueError("episode did not consume its exact reservation")
        result = dict(world=world, mode=mode, iteration=job["iteration"],
                      J600=-math.fsum(rewards), episode_return=math.fsum(rewards),
                      decisions=DECISIONS, actual_sim_s=backend.actual, components=metrics,
                      elapsed_s=time.monotonic() - begin)
        shared.storage.append(root / "result.jsonl", result)
    finally:
        if backend is not None:
            backend.close()
    if not capability.finished:
        raise ValueError("runtime did not issue terminal actual accounting")
    return dict(frames=frames, result=result)


def _worker_main(connection, shared_root, sources):
    try:
        configure_cpu()
        shared = load_shared(shared_root)
        assert_sources(shared, sources)
        validate_allocation(shared)
        connection.send(dict(event="ready"))
        while True:
            job = connection.recv()
            if job is None:
                return
            seed = job["reservation"]["seed"]
            try:
                assert_sources(shared, sources)
                output = run_episode(shared, job, connection.send)
                connection.send(dict(event="result", seed=seed, **output))
            except BaseException as error:
                connection.send(dict(event="error", seed=seed,
                                     error=f"{type(error).__name__}: {error}"))
                return
    except BaseException as error:
        connection.send(dict(event="startup_error", error=f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


@dataclass
class Worker:
    process: object
    connection: object
    seed: int | None = None
    alive: bool = True


class Workers:
    """Spawned, persistent, independent runtime processes; never signal PIDs."""
    def __init__(self, count, shared_root, sources):
        if type(count) is not int or not 1 <= count <= MAX_WORKERS:
            raise ValueError("workers must be in [1,12]")
        self.workers = []
        self.limit, self.shared_root, self.sources = count, shared_root, sources
        self.context = mp.get_context("spawn")
        try:
            # A single existing CPU/nice19 worker runs BOTH mechanics canaries.
            self._start_worker()
        except BaseException:
            self.close()
            raise

    def _start_worker(self):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_worker_main,
                                       args=(child, self.shared_root, self.sources))
        process.start()
        child.close()
        worker = Worker(process, parent)
        self.workers.append(worker)
        event = parent.recv()
        if event["event"] != "ready":
            raise RuntimeError(event.get("error", "worker failed startup"))

    def expand(self, coordinator):
        """Durable successful smoke receipts, never a caller-supplied pass flag."""
        if coordinator.failed or any(
                seed not in coordinator.records
                or not coordinator.records[seed]["complete"]
                or coordinator.records[seed]["actual"] != DURATION
                or coordinator.records[seed]["reservation"]["mode"] != "smoke"
                for seed in SMOKE_SEEDS):
            raise RuntimeError("both single-worker smoke episodes must succeed before expansion")
        if any(worker.seed is not None for worker in self.workers):
            raise RuntimeError("cannot expand while episodes are active")
        while len(self.workers) < self.limit:
            self._start_worker()

    def close(self, coordinator=None):
        if getattr(self, "closed", False):
            return
        self.closed = True
        # Drain in-flight messages before joining, so a large trajectory cannot
        # block a worker on a full pipe during cooperative shutdown.
        for worker in self.workers:
            if worker.alive or worker.seed is not None:
                if worker.process.is_alive():
                    try:
                        worker.connection.send(None)
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                try:
                    while True:
                        ready = wait([worker.connection, worker.process.sentinel])
                        if worker.connection in ready:
                            try:
                                event = worker.connection.recv()
                                if coordinator is not None and worker.seed is not None:
                                    self._shutdown_receipt(worker, event, coordinator)
                            except EOFError:
                                break
                        else:
                            break
                except (BrokenPipeError, EOFError, OSError):
                    pass
            worker.process.join()
            if coordinator is not None and worker.seed is not None:
                coordinator.fail(worker.seed, "worker exited without terminal receipt during shutdown")
                worker.seed = None
            worker.connection.close()

    @staticmethod
    def _shutdown_receipt(worker, event, coordinator):
        seed = worker.seed
        if event.get("seed") != seed:
            return
        if event["event"] == "actual":
            coordinator.actual(seed, event["actual_sim_s"])
        else:
            # This path is entered only after an exceptional interruption of
            # batch(); keep the actual receipt but do not call it resumable.
            coordinator.fail(seed, event.get("error", "supervisor interrupted during episode"))
            worker.seed = None

    def batch(self, worlds_, mode, base_job, coordinator, shared, sources, deadline, stop_path):
        worlds_, outputs = list(worlds_), {}
        cursor, failure = 0, None
        exhausted = not worlds_
        try:
            while True:
                active = [w for w in self.workers if w.seed is not None]
                if time.monotonic() >= deadline or stop_path.exists():
                    failure = failure or "wall limit or cooperative stop"
                # Collect every observable event before authorizing more work.
                ready = wait([w.connection for w in active], timeout=0) if active else []
                awaiting_result = any(coordinator.records[w.seed]["actual"] is not None for w in active)
                if not ready and not failure and not awaiting_result and not exhausted:
                    try:
                        assert_sources(shared, sources)
                        for worker in self.workers:
                            if worker.seed is not None:
                                continue
                            if not worker.alive or not worker.process.is_alive():
                                raise RuntimeError("worker exited; dispatch stopped")
                            if time.monotonic() >= deadline or stop_path.exists():
                                raise RuntimeError("wall limit or cooperative stop")
                            # A failed peer can report while earlier jobs are
                            # being reserved. Recheck pipes before each dispatch.
                            if any(w.seed is not None and w.connection.poll() for w in self.workers):
                                break
                            if cursor == len(worlds_):
                                exhausted = True
                                break
                            pending = sum(w.seed is not None for w in self.workers)
                            if not resource_admitted(stop_path.parent, shared.protocol, pending):
                                break
                            if time.monotonic() >= deadline or stop_path.exists():
                                raise RuntimeError("wall limit or cooperative stop")
                            world = worlds_[cursor]
                            reservation = coordinator.reserve(world, mode)
                            worker.seed = world["seed"]
                            try:
                                worker.connection.send(dict(base_job, reservation=reservation))
                            except Exception as error:
                                coordinator.fail(worker.seed, f"dispatch failed: {error}")
                                worker.seed = None
                                raise
                            cursor += 1
                            exhausted = cursor == len(worlds_)
                    except Exception as error:
                        failure = str(error)
                active = [w for w in self.workers if w.seed is not None]
                if not active:
                    if failure:
                        raise RuntimeError(failure)
                    if exhausted:
                        return [outputs[world["seed"]] for world in worlds_]
                    time.sleep(min(RESOURCE_WAIT_SECONDS, max(0., deadline - time.monotonic())))
                    continue
                ready = wait([w.connection for w in active] + [w.process.sentinel for w in active],
                             timeout=1.)
                for worker in active:
                    if worker.connection not in ready and worker.process.sentinel not in ready:
                        continue
                    seed = worker.seed
                    try:
                        event = worker.connection.recv()
                    except (EOFError, OSError):
                        event = dict(event="error", seed=seed, error="worker exited without receipt")
                    try:
                        if event.get("seed") != seed:
                            raise ValueError("worker receipt seed mismatch")
                        if event["event"] == "actual":
                            coordinator.actual(seed, event["actual_sim_s"])
                            if event["actual_sim_s"] != DURATION:
                                failure = failure or "runtime closed before 900 seconds; dispatch stopped"
                            continue
                        if event["event"] != "result":
                            raise RuntimeError(event.get("error", "invalid worker receipt"))
                        validate_result(event, coordinator.records[seed]["reservation"])
                        coordinator.complete(seed)
                        outputs[seed] = event
                    except Exception as error:
                        failure = failure or str(error)
                        coordinator.fail(seed, str(error))
                    worker.seed = None
                    if event["event"] != "result":
                        worker.alive = False
        except BaseException:
            # Preserve late actual receipts on interruption, without authorizing
            # any more work. close is idempotent for the caller's finally block.
            self.close(coordinator)
            raise


def validate_result(event, reservation):
    result, frames = event["result"], event["frames"]
    if (result["world"] != reservation["world"] or result["mode"] != reservation["mode"]
            or result["decisions"] != DECISIONS or result["actual_sim_s"] != DURATION
            or not math.isfinite(result["J600"]) or not math.isfinite(result["episode_return"])
            or result["J600"] != -result["episode_return"]):
        raise ValueError("invalid complete-episode receipt")
    expected = DECISIONS if reservation["mode"] == "training" else 0
    if len(frames) != expected:
        raise ValueError("incomplete training trajectory")
    if frames and not math.isclose(math.fsum(t["reward"] for t in frames), result["episode_return"]):
        raise ValueError("trajectory/team-return mismatch")


def decode_trajectory(shared, frames):
    import torch
    return [shared.learning.Transition(
        shared.observations.Observation(tuple(t["keys"]), tuple(t["actors"]), tuple(t["shared"])),
        t["actions"], torch.tensor(t["old_logp"], dtype=torch.float32), t["reward"])
        for t in frames]


def return_statistics(outputs):
    stats = {}
    for field in ("J600", "episode_return"):
        values = [row["result"][field] for row in outputs]
        mean = math.fsum(values) / len(values)
        stats.update({f"{field}_mean": mean, f"{field}_min": min(values),
                      f"{field}_max": max(values),
                      f"{field}_std": math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))})
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--c1-root", required=True)
    parser.add_argument("--integration", required=True)
    parser.add_argument("--design-sha", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, MAX_WORKERS + 1))
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--evaluation-contract")
    parser.add_argument("--evaluation-contract-sha")
    args = parser.parse_args(argv)
    configure_cpu()
    shared = load_shared(args.shared_root)
    sources = source_receipt(shared)
    frames = validate_environment(shared)
    validate_allocation(shared)
    if not shared.protocol.approval_line(args.registration):
        raise PermissionError("shared MAIN approval absent")
    if not shared.protocol.audit_registration(args.registration, variant=VARIANT,
                                              design_sha=args.design_sha)["passed"]:
        raise ValueError("shared seed allocation conflict")
    shared.integration.load(args.integration, args.registration, args.c1_root, args.design_sha,
                            variant=VARIANT)
    bindings = dict(integration_sha256=shared.protocol.digest(args.integration),
                    design_sha256=args.design_sha)
    if bool(args.evaluation_contract) != bool(args.evaluation_contract_sha):
        raise ValueError("evaluation requires both explicit contract and SHA256")
    if args.mode == "evaluate" and not args.evaluation_contract:
        raise ValueError("evaluation panel pending freeze; explicit contract required")
    panel, contract_digest = (evaluation_worlds(shared, args.evaluation_contract,
                                               args.evaluation_contract_sha, args.registration)
                              if args.evaluation_contract else (None, None))
    validate_learning(shared, frames)
    import numpy as np
    import torch
    random.seed(OPTIMIZATION_SEED)
    np.random.seed(OPTIMIZATION_SEED)
    torch.manual_seed(OPTIMIZATION_SEED)
    out = shared.protocol.safe_path(args.out)
    start = time.monotonic()
    if args.mode == "train":
        out.mkdir(parents=True, exist_ok=False)
        exclusive_json(out / "OWNER.json", dict(pid=os.getpid(), variant=VARIANT,
                       registration_sha256=shared.protocol.digest(args.registration),
                       sources=sources, bindings=bindings, workers=args.workers,
                       optimization_seed=OPTIMIZATION_SEED))
    # Persistent exclusive phase ownership, including inline evaluation.
    exclusive_json(out / f"{args.mode.upper()}_OWNER.json", dict(pid=os.getpid()))
    if args.mode == "train" and panel is not None:
        exclusive_json(out / "EVALUATE_OWNER.json", dict(pid=os.getpid()))
    coordinator = Coordinator(out, shared, panel=tuple(w["seed"] for w in panel) if panel else ())
    base = dict(out=str(out), registration=str(shared.protocol.safe_path(args.registration)),
                c1_root=str(shared.protocol.safe_path(args.c1_root)), sources=sources,
                integration=str(shared.protocol.safe_path(args.integration)), design_sha=args.design_sha,
                bindings=bindings)
    workers = Workers(args.workers, args.shared_root, sources)
    try:
        if args.mode == "train":
            model = shared.learning.Selector().cpu()
            updater = shared.learning.Updater(model)
            checkpoint = save_checkpoint(out, 0, model, updater, sources, bindings)
            base.update(checkpoint=str(checkpoint), checkpoint_sha256=shared.protocol.digest(checkpoint),
                        iteration=0)
            workers.batch([dict(seed=s, penetration=.4, demand=2700) for s in SMOKE_SEEDS],
                          "smoke", base, coordinator, shared, sources,
                           start + WALL_SECONDS, out / "STOP_REQUESTED")
            workers.expand(coordinator)
            for iteration in range(ITERATIONS):
                begin = time.monotonic()
                outputs = workers.batch(worlds(iteration, shared.protocol), "training", base,
                                        coordinator, shared, sources, start + WALL_SECONDS,
                                        out / "STOP_REQUESTED")
                simulation_wall = time.monotonic() - begin
                stats = updater.update([decode_trajectory(shared, row["frames"]) for row in outputs])
                checkpoint = save_checkpoint(out, iteration + 1, model, updater, sources, bindings)
                elapsed = time.monotonic() - begin
                shared.storage.append(out / "training.jsonl", dict(
                    iteration=iteration + 1, **stats, **return_statistics(outputs),
                    simulation_wall_s=simulation_wall, total_wall_s=elapsed,
                    sim_s_per_hour=EPISODES * DURATION / elapsed * 3600,
                    charged_sim_s=coordinator.charged, actual_sim_s=coordinator.actual_sim_s,
                    checkpoint_sha256=shared.protocol.digest(checkpoint)))
                base.update(checkpoint=str(checkpoint), checkpoint_sha256=shared.protocol.digest(checkpoint),
                            iteration=iteration + 1)
            assert_sources(shared, sources)
            exclusive_json(out / "FROZEN_FINAL.json", dict(iteration=ITERATIONS,
                           checkpoint_sha256=shared.protocol.digest(checkpoint), sources=sources,
                           bindings=bindings))
        if panel is not None:
            final = checkpoint_path(out, ITERATIONS)
            frozen = json.loads(shared.protocol.safe_path(out / "FROZEN_FINAL.json").read_text(encoding="utf-8"))
            if (frozen["iteration"] != ITERATIONS or frozen["sources"] != sources
                    or frozen["checkpoint_sha256"] != shared.protocol.digest(final)
                    or frozen["bindings"] != bindings):
                raise ValueError("frozen final169 contract mismatch")
            exclusive_json(out / "EVALUATION_CONTRACT.json", dict(
                contract="S-C1SWITCH2", sha256=contract_digest, worlds=panel,
                checkpoint_sha256=frozen["checkpoint_sha256"]))
            base.update(checkpoint=str(final), checkpoint_sha256=frozen["checkpoint_sha256"],
                        iteration=ITERATIONS)
            workers.expand(coordinator)
            outputs = workers.batch(panel, "evaluation", base, coordinator, shared, sources,
                                    start + WALL_SECONDS, out / "STOP_REQUESTED")
            shared.storage.append(out / "evaluation.jsonl", dict(
                iteration=ITERATIONS, contract_sha256=contract_digest,
                **return_statistics(outputs), charged_sim_s=coordinator.charged,
                actual_sim_s=coordinator.actual_sim_s))
    finally:
        workers.close(coordinator)


if __name__ == "__main__":
    main()
