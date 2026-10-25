"""Record/compare every forward and short-rollout tensor against the R13C tip.

Run record twice in fresh CPU processes with --repo BASE and --repo CANDIDATE,
then --compare BASE.pt CANDIDATE.pt. Uses four predetermined TRAIN states,
both CRN samples and twelve stop-time integration steps; never fits anything.
"""
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True

import diag_r13y_offline as protocol  # noqa: E402


def tensors(value, prefix, out):
    import torch
    if torch.is_tensor(value):
        out[prefix] = value.detach().cpu().clone()
    elif isinstance(value, dict):
        for key, item in value.items():
            tensors(item, prefix + "/" + str(key), out)
    elif isinstance(value, (list, tuple)):
        for key, item in enumerate(value):
            tensors(item, prefix + "/" + str(key), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.compare:
        a, b = [torch.load(protocol.guard_path(p), map_location="cpu", weights_only=True)
                for p in args.compare]
        assert a.keys() == b.keys(), "tensor key mismatch"
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
            # Compare representations too (+0/-0 and NaN payloads), not just values.
            left = a[key].contiguous().reshape(-1).view(torch.uint8)
            right = b[key].contiguous().reshape(-1).view(torch.uint8)
            assert torch.equal(left, right), key
        print(f"PASS: {len(a)} tensors bit-identical; forward + 8 short rollouts + weights")
        return
    out = protocol.guard_path(args.out)
    assert out.parent.is_dir() and not out.exists()
    protocol.install_access_guard(out.parent)
    repo = protocol.guard_path(args.repo)
    fw = protocol.load_worker(repo, "/root/autodl-tmp/c4/v5/diag_blowup")
    records = protocol.read_states(
        "/root/autodl-tmp/c4/v5/chk_r10b/states192.json.gz",
        "6b7d15f6081ad703339c9e149b217cf382e8cf0b8aa83fb3ab13b786df3a37ec")
    label = "r13yOff_Tasis_Dasis_Con"
    fw.init_arms([(label, "/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round12c/wm_s.pt")])
    predictor, cap = fw.G["p"][label]
    assert not getattr(predictor.model, "mainline_merge_yield_learnable", False)
    captured = {}
    tensors(predictor.model.state_dict(), "weights", captured)
    original = predictor.model.forward
    current = {}

    def forward(*a, **kw):
        result = original(*a, **kw)
        tensors(result, f"{current['prefix']}/forward/{current['step']}", captured)
        current["step"] += 1
        return result

    predictor.model.forward = forward
    with torch.no_grad():
        for idx in (0, 16, 96, 112):
            rec = dict(records[idx], _fact_idx=idx // 4)
            for sample in (0, 1):
                current.update(prefix=f"state{idx}/sample{sample}", step=0)
                fw.roll_once(label, rec, sample, 12)
                tensors(cap["last"], current["prefix"] + "/rollout", captured)
    torch.save(captured, out)
    print(f"recorded {len(captured)} tensors from {repo}")


if __name__ == "__main__":
    main()
