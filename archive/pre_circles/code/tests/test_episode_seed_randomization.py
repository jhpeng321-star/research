"""Per-episode seed randomization + training episode-length default (Task 11).

Training resets (``reset()`` with no seed) must draw fresh CAV-selection and
HDV-type seeds from a master RNG so consecutive episodes see different CAV
sets / driver mixes, while the whole run stays reproducible from
``config["cav_selection_seed"]``. Explicit ``reset(seed=...)`` keeps the
legacy deterministic behavior used by paired evaluation.
"""

import ast
from pathlib import Path

from env.core import TrafficShepherdMultiAgentEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ENV_CONFIG = {
    "sumo_binary": "sumo",
    "sumo_config": "osm.light.sumocfg",
    "cav_penetration": 0.20,
    "max_episode_steps": 400,
    "simulation_step": 1.0,
    "cav_warmup_steps": 300,
}

_COLLECT_STEPS = 80


def _collect_cav_ids(env, seed=None, steps=_COLLECT_STEPS):
    """Reset and roll a few non-intervention steps; return CAV ids seen.

    SUMO runs deterministically here (``sumo_seed`` None -> SUMO default), so
    the departed vehicle sequence is identical across episodes and any
    difference in the selected CAV set comes from the episode seed.
    """
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=seed)
    seen = set(info["active_cavs"])
    for _ in range(steps):
        action_dict = {cav_id: 0 for cav_id in obs.keys()}
        obs, _, terminated, truncated, info = env.step(action_dict)
        seen.update(info["active_cavs"])
        if terminated["__all__"] or truncated["__all__"]:
            break
    return seen


def test_successive_resets_randomize_cav_selection():
    """Two reset() calls without a seed must produce different CAV sets."""
    env = TrafficShepherdMultiAgentEnv(dict(_ENV_CONFIG))
    try:
        first = _collect_cav_ids(env)
        second = _collect_cav_ids(env)
    finally:
        env.close()
    assert first, "expected at least one CAV in episode 1"
    assert second, "expected at least one CAV in episode 2"
    assert first != second, (
        "consecutive unseeded resets selected identical CAV sets; "
        "per-episode seed randomization is missing"
    )


def test_master_seed_reproducible_episode_sequence():
    """Same construction seed -> same per-episode seed sequence across envs."""
    env_a = TrafficShepherdMultiAgentEnv(dict(_ENV_CONFIG))
    env_b = TrafficShepherdMultiAgentEnv(dict(_ENV_CONFIG))
    try:
        a_ep1 = _collect_cav_ids(env_a)
        a_ep2 = _collect_cav_ids(env_a)
        b_ep1 = _collect_cav_ids(env_b)
        b_ep2 = _collect_cav_ids(env_b)
    finally:
        env_a.close()
        env_b.close()
    assert a_ep1 == b_ep1, "episode 1 diverged between identically seeded envs"
    assert a_ep2 == b_ep2, "episode 2 diverged between identically seeded envs"
    assert a_ep1 != a_ep2, "episode seeds did not advance within one env"


def test_explicit_seed_keeps_eval_determinism():
    """reset(seed=42) must stay deterministic (paired evaluation path)."""
    env = TrafficShepherdMultiAgentEnv(dict(_ENV_CONFIG))
    try:
        first = _collect_cav_ids(env, seed=42)
        # An unseeded episode in between must not disturb the explicit path.
        _collect_cav_ids(env)
        second = _collect_cav_ids(env, seed=42)
    finally:
        env.close()
    assert first, "expected at least one CAV with explicit seed"
    assert first == second, "explicit-seed resets are not deterministic"


def _master_stream(env_config, n=4):
    """First n draws of the env's master episode-seed RNG (no SUMO needed)."""
    env = TrafficShepherdMultiAgentEnv(dict(env_config))
    try:
        return env._episode_seed_rng.integers(0, 2**31 - 1, size=n).tolist()
    finally:
        env.close()


def test_worker_index_diversifies_master_rng_stream():
    """worker/vector index diversify the master RNG; missing keys fall back
    to the legacy base-seed stream (single-env / direct construction)."""
    base = _master_stream(_ENV_CONFIG)
    w0 = _master_stream({**_ENV_CONFIG, "worker_index": 0, "vector_index": 0})
    w1 = _master_stream({**_ENV_CONFIG, "worker_index": 1, "vector_index": 0})
    w2 = _master_stream({**_ENV_CONFIG, "worker_index": 2, "vector_index": 0})
    w1v1 = _master_stream({**_ENV_CONFIG, "worker_index": 1, "vector_index": 1})
    assert base == w0, "index 0/0 must match the legacy base-seed stream"
    assert w1 != base and w2 != base, "worker index did not change the stream"
    assert w1 != w2, "workers 1 and 2 share the same master RNG stream"
    assert w1v1 != w1, "vector index did not change the stream"
    assert _master_stream({**_ENV_CONFIG, "worker_index": 1, "vector_index": 0}) == w1, (
        "same worker index must reproduce the same stream"
    )


def test_worker_index_diversifies_cav_selection():
    """Two env runners with different worker_index select different CAV sets
    on unseeded resets (training diversity across workers)."""
    env_w1 = TrafficShepherdMultiAgentEnv({**_ENV_CONFIG, "worker_index": 1})
    env_w2 = TrafficShepherdMultiAgentEnv({**_ENV_CONFIG, "worker_index": 2})
    try:
        cavs_w1 = _collect_cav_ids(env_w1)
        cavs_w2 = _collect_cav_ids(env_w2)
    finally:
        env_w1.close()
        env_w2.close()
    assert cavs_w1 and cavs_w2, "expected CAVs in both worker envs"
    assert cavs_w1 != cavs_w2, (
        "different worker_index selected identical CAV sets; "
        "master RNG is not diversified across env workers"
    )


def test_same_worker_index_keeps_episode_sequence_reproducible():
    """Same worker_index + same base seed -> identical episode sequences."""
    env_a = TrafficShepherdMultiAgentEnv({**_ENV_CONFIG, "worker_index": 3})
    env_b = TrafficShepherdMultiAgentEnv({**_ENV_CONFIG, "worker_index": 3})
    try:
        a_ep1 = _collect_cav_ids(env_a)
        a_ep2 = _collect_cav_ids(env_a)
        b_ep1 = _collect_cav_ids(env_b)
        b_ep2 = _collect_cav_ids(env_b)
    finally:
        env_a.close()
        env_b.close()
    assert a_ep1 == b_ep1, "episode 1 diverged between same-worker envs"
    assert a_ep2 == b_ep2, "episode 2 diverged between same-worker envs"
    assert a_ep1 != a_ep2, "episode seeds must still advance within one env"


def test_rllib_wrapper_forwards_env_context_worker_index():
    """Old-stack RLlib hands the env creator an EnvContext whose worker_index
    is an *attribute* (not a dict key); the wrapper must forward it so the
    inner env diversifies its master RNG."""
    from ray.rllib.env.env_context import EnvContext

    from env.rllib_wrapper import TrafficShepherdRLlibEnv

    ctx = EnvContext(dict(_ENV_CONFIG), worker_index=7, vector_index=0, num_workers=12)
    env = TrafficShepherdRLlibEnv(ctx)
    try:
        assert env._env.config["worker_index"] == 7
        stream = env._env._episode_seed_rng.integers(0, 2**31 - 1, size=4).tolist()
    finally:
        env.close()
    assert stream == _master_stream({**_ENV_CONFIG, "worker_index": 7})


def test_train_default_max_episode_steps_is_1200():
    """train.py --max-episode-steps must default to 1200 (900s window + clearing)."""
    tree = ast.parse((PROJECT_ROOT / "train.py").read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        flags = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if "--max-episode-steps" not in flags:
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                found = kw.value.value
    assert found == 1200, f"--max-episode-steps default is {found!r}, expected 1200"
