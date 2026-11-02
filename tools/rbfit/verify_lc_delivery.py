"""Verify a published LC export through the actual UNC sampler, without tapes."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=Path, required=True)
    parser.add_argument("--idm", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verification = json.loads((args.lane / "VERIFIED.json").read_bytes())
    assert verification["status"] == "PASS"
    for name, expected in verification["artifact_sha256"].items():
        assert hashlib.sha256((args.lane / name).read_bytes()).hexdigest() == expected
    population = json.loads((args.lane / "lc2013_population.json").read_bytes())
    posterior = json.loads((args.lane / "lc2013_posteriors.json").read_bytes())
    assert population["fit_worlds"] == list(range(969400, 969432))
    assert population["fit_interval_s"] == [240, 300]
    assert posterior["conditioning_end_s"] == 300
    assert posterior["shrinkage_already_applied"]
    spec = importlib.util.spec_from_file_location("delivery_adapter", args.adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = module.load_boundary_fitted_state(args.idm, lane_directory=args.lane)
    reader.check_time(300)
    order = population["parameter_order"]
    support = {tuple(row) for row in population["support"]}
    generator = torch.Generator().manual_seed(190)
    for key in posterior["drivers"]:
        world, vid = map(int, key.split(":"))
        assert 969400 <= world <= 969431
        draw = reader.draw(world, vid, lane=True, generator=generator)
        assert tuple(draw[name] for name in order) in support
    for vid in (None, 999999):
        a = reader.draw(969400, vid, lane=True, generator=torch.Generator().manual_seed(91))
        b = reader.draw(969400, vid, lane=True, generator=torch.Generator().manual_seed(91))
        assert a == b and tuple(a[name] for name in order) in support

    from wms.control.hidden_state import plan_hidden_state_futures
    from wms.model.driver_encoder import PARAMETER_NAMES
    points = torch.tensor([30., 1., 2., 1., 2., .4, .2, 8., 1.])
    parameters = {name: points[i].expand(2, 5).clone()
                  for i, name in enumerate(PARAMETER_NAMES)}

    class Model:
        physics = SimpleNamespace(dt_s=.5, entry_learned=False)

        def encode_drivers(self, state, **kwargs):
            return {}, None, parameters

    vids = [int(key.split(":")[1]) for key in posterior["drivers"]
            if key.startswith("969400:")][:3]
    ids = torch.tensor([vids + [999999, -1]]).expand(2, -1)
    state = {"present": torch.tensor([[True, True, True, True, False]]).expand(2, -1),
             "is_cav": torch.tensor([[False, False, False, True, False]]).expand(2, -1)}
    options = dict(future_sampling="hidden_state", n_samples=8, seed=190,
                   row_keys=[969400, 969400], vehicle_ids=ids, t0_s=300, fitted_state=reader)
    rng = torch.get_rng_state().clone()
    futures = plan_hidden_state_futures(Model(), state, {}, **options)
    replay = plan_hidden_state_futures(Model(), state, {}, **options)
    assert torch.equal(rng, torch.get_rng_state())
    for a, b in zip(futures, replay):
        for name, values in a.kwargs["driver_params"].items():
            assert torch.equal(values, b.kwargs["driver_params"][name])
            assert torch.equal(values[0], values[1])
            baseline = parameters[name][:, 3:] if name in parameters else torch.ones(2, 2)
            assert torch.equal(values[:, 3:], baseline)
    scores = {}
    for name in ("raw_physics_population", "raw_physics_shrunk_posterior", "raw_physics_source_default"):
        raw = verification["aggregate"][name]
        matched = raw["expected_correct_direction_event_count"]
        scores[name] = dict(raw, ratio_of_expectations_precision=matched / raw["expected_event_count"],
                            exact_directed_recall=matched / raw["events"])
    keep_right = order.index("lcKeepRight")
    zero_mass = math.fsum(w for row, w in zip(population["support"], population["weights"])
                         if row[keep_right] == 0)
    result = {"status": "PASS", "drivers_drawn": len(posterior["drivers"]),
              "paired_replayed_futures": len(futures) + len(replay),
              "global_rng_unchanged": True, "population_keep_right_zero_mass": zero_mass,
              "scores": scores, "adopted": False,
              "qualification": "Conditional FIT engineering/sampler only; no autonomous traffic gate",
              "verification_sha256": hashlib.sha256((args.lane / "VERIFIED.json").read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
