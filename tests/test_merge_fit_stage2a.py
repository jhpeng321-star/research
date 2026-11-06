"""F1/F2 integration and audit isolation, with no likelihood support floors."""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from tools import merge_fit_stage2a as fit
from tools import merge_fit_v2 as old
from tools import r9_exact as exact
from tools import r9_fit_exact as heads


def initial():
    return dict(s0=2., T=.8, b=2., tau_a=1., temperature=1., sigma=.3)


def sequence(world="w1", **overrides):
    n = 5
    features = dict(gap_l=[10., 11., 9., 12., 13.], gap_f=[12.]*n,
                    v=[4.]*n, v_l=[4.]*n, v_f=[4.]*n, a=[0.]*n,
                    a_l=[0.]*n, a_f=[0.]*n, has_l=[True]*n, has_f=[True]*n)
    return dict(world=world, source="synthetic", episode=world+"e", vehicle=1,
                slot=0, entry_row=1, event_row=6, legacy=features, updated=copy.deepcopy(features),
                physical=[True]*n, physical_reasons=["ok"]*n, legacy_support=[True]*n,
                event=True, terminal_kind="event", unknown_age=False, queue="front", demand=1000,
                dt=.5, times=[.5*(i+1) for i in range(n)], observed_s=2.5, horizon_s=300.,
                **overrides)


def test_f1_normalizes_and_retains_single_persistent_latent():
    c = torch.tensor([[-1., -.2, .3, -.5, .8]], dtype=torch.float64, requires_grad=True)
    tau, w = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (.7, .2))
    dist = fit.hybrid(c, tau, w, 64)
    torch.testing.assert_close(dist["log_mass"].exp().sum()+dist["log_survival"][0, -1].exp(),
                               torch.tensor(1., dtype=torch.float64), atol=1e-12, rtol=0)
    (-dist["log_mass"][0, -1]).backward()
    assert torch.isfinite(c.grad).all() and tau.grad != 0 and w.grad != 0
    z, weights = old.logistic_quadrature(64)
    hazards = torch.sigmoid((c.detach().unsqueeze(1)+tau.detach()*z.view(1, -1, 1))/w.detach())
    wrong_survival = (1-(hazards*weights.view(1, -1, 1)).sum(1)).prod(-1)
    assert not torch.allclose(wrong_survival, dist["log_survival"][:, -1].exp())


def test_f1_w_zero_limit_and_deterministic_atoms():
    c = torch.tensor([[-math.inf, 0., 0., -1., math.inf]], dtype=torch.float64)
    finite = torch.isfinite(c)
    dist = fit.hybrid(c, torch.tensor(.7), torch.tensor(1e-9), 64)
    strict = old.first_passage(c, torch.tensor(.7))
    torch.testing.assert_close(dist["log_mass"].exp(), strict["log_mass"].exp(), atol=1e-6, rtol=0)
    torch.testing.assert_close(dist["cdf"], strict["cdf"], atol=1e-6, rtol=0)
    assert finite.any() and torch.isneginf(dist["log_survival"][0, -1])
    later = fit.hybrid(torch.tensor([[math.inf, 1.]], dtype=torch.float64), torch.tensor(.3), torch.tensor(.2))
    assert later["log_mass"][0, 0] == 0 and torch.isneginf(later["log_mass"][0, 1])


def test_f1_veto_has_exact_zero_mass_and_no_gradient_floor():
    dist = fit.hybrid(torch.tensor([[1000., 0.]], dtype=torch.float64), torch.tensor(.3),
                      torch.tensor(.2), eligible=torch.tensor([[False, True]]))
    assert torch.isneginf(dist["log_mass"][0, 0])
    assert dist["cdf"][0, 0] == 0
    assert torch.isfinite(dist["log_mass"][0, 1])


@pytest.mark.parametrize("kind", ["F1", "P1", "P1+P2", "P1-P3", "M0-approx"])
def test_f2_prefix_conditions_and_log_score_telescopes(kind):
    model = fit.Candidate(kind, initial())
    s = sequence()
    whole = model.distribution(fit.pack([s]), 64)
    s["score_start"] = 2
    conditional = model.distribution(fit.pack([s]), 64)
    den = whole["log_survival"][0, 1]
    torch.testing.assert_close(conditional["log_mass"][0, 2:]+den, whole["log_mass"][0, 2:])
    torch.testing.assert_close(conditional["log_survival"][0, 2:]+den, whole["log_survival"][0, 2:])
    torch.testing.assert_close(conditional["log_mass"][0, 2:].exp().sum()+conditional["log_survival"][0, -1].exp(),
                               torch.tensor(1., dtype=torch.float64), atol=1e-12, rtol=0)


def test_unknown_history_zero_conditioning_and_exits_never_reset_prior():
    model = fit.Candidate("F1", initial())
    s = sequence()
    s["unknown_age"] = True
    assert torch.isinf(model.nll(fit.pack([s]))).all()
    s["unknown_age"] = False
    s["terminal_kind"] = "junction_exit"
    assert torch.isinf(model.nll(fit.pack([s]))).all()
    s["terminal_kind"] = "event"
    s["updated"]["has_l"] = [False]*5
    s["updated"]["has_f"] = [False]*5
    s["score_start"] = 1
    assert torch.isinf(model.nll(fit.pack([s]))).all()


def test_f1_penalty_and_grouped_cartesian_selection():
    args = SimpleNamespace(iters=2, lr=.01, n_quad=16, inner_folds=2, shrinkage=[0., .01])
    seqs = [sequence("w"+str(i)) for i in range(4)]
    model, report = fit.select("F1", initial(), None, seqs, args)
    assert report["selection_status"] == "ok"
    assert len(report["inner"]) == 4
    assert report["final_fit"]["updates"] == 2
    assert model.params()["w"] > 0
    for trial in report["inner"]:
        groups = [set(fold["worlds"]) for fold in trial["folds"]]
        assert groups[0].isdisjoint(groups[1])
    assert (0., 0.) in [(r["lambda_tau"], r["lambda_w"]) for r in report["inner"]]


@pytest.mark.parametrize('pair', [(0., 0.), (.002, 0.)])
@pytest.mark.parametrize('bucket_sizes', [(1, 2), (3,)])
def test_tf_objective_optimizer_scales_batches_gradients_and_final_selection(monkeypatch, pair,
                                                                           bucket_sizes):
    class QuadraticTF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

        def nll(self, batch, n_quad):
            return (self.value - batch).square()

        def params(self):
            return {'tau': self.value}

    model = QuadraticTF()
    monkeypatch.setattr(fit, 'make_model', lambda *args: model)
    # Unequal batches catch an accidental mean reduction or per-batch denominator.
    packed = torch.tensor([0., 1., 3.], dtype=torch.float64).split(bucket_sizes)
    monkeypatch.setattr(fit, 'batches', lambda sequences: packed)
    start_nll = sum(float(model.nll(b, 2).sum().detach()) for b in packed)
    expected_start = (0.6 / 395.2) * start_nll / 3 + pair[0] * 4.
    expected_grad = (0.6 / 395.2) * 4. / 3 + pair[0] * 4.
    fitted, report = fit.optimize('F1', None, None, [None]*3,
                                  SimpleNamespace(iters=1, lr=.01, n_quad=2), pair)
    assert report['status'] == 'fitted'
    assert report['trace'][0]['sum_objective'] == pytest.approx(expected_start)
    assert report['trace'][0]['gradient_norms']['value'] == pytest.approx(expected_grad)
    final_nll = sum(float(fitted.nll(b, 2).sum().detach()) for b in packed)
    expected_final = (0.6 / 395.2) * final_nll / 3 + pair[0] * float(fitted.value.detach())**2
    assert report['sum_objective'] == pytest.approx(expected_final)
    assert report['sum_objective'] < expected_start
    assert fitted.value < 2.  # the post-update score must use the same normalization


def test_reconciles_one_denominator_and_refuses_duplicate_exposure():
    seqs = [sequence()]
    excluded = dict(sequence("w2"), reason="CAV_not_supervised_neighbor_retained")
    report = fit.reconcile(seqs, [{"exclusions": [excluded]}])
    assert report["original_observed_risk_identities"] == report["retained"]+report["excluded"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        fit.reconcile(seqs*2, [])


def test_paired_results_require_identical_identity_order():
    args = SimpleNamespace(n_quad=16, bootstrap=20, seed=73)
    ev = fit.evaluate(fit.Candidate("F1", initial()), [sequence("w1"), sequence("w2")], args)
    other = copy.deepcopy(ev)
    other["cars"].reverse()
    with pytest.raises(ValueError, match="identity/order"):
        fit.paired(ev, other, args)


def test_f1_save_reload_preserves_learned_width_and_conditional_law(tmp_path):
    model = fit.Candidate("F1", initial())
    with torch.no_grad():
        model.base.raw["w"].add_(.4)
        model.base.raw["tau"].sub_(.2)
    seq = sequence()
    seq["score_start"] = 2
    batch = fit.pack([seq])
    path = tmp_path/"f1.pt"
    torch.save(model.state_dict(), path)
    clone = fit.Candidate("F1", initial())
    clone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    torch.testing.assert_close(model.params()["w"], clone.params()["w"], atol=1e-5, rtol=0)
    torch.testing.assert_close(model.params()["tau"], clone.params()["tau"], atol=1e-5, rtol=0)
    for key, value in model.distribution(batch).items():
        torch.testing.assert_close(value, clone.distribution(batch)[key], atol=1e-6, rtol=0)


def test_default_entrypoint_is_stage2a_and_smoke_cannot_open_audit(monkeypatch):
    called = []
    monkeypatch.setattr(fit, "main", lambda argv: called.append(argv) or 0)
    assert old.main(["--source", "native=dataset"]) == 0
    assert called == [["--source", "native=dataset"]]


def test_audit_ledger_refuses_reuse_before_checkpoint_read(tmp_path, monkeypatch):
    # tmp_path is forced inside workspace by the required pytest --basetemp.
    assert tmp_path.is_relative_to(fit.ROOT)
    ledger = tmp_path/"r9_stage2a_audit_use.json"
    ledger.write_text("{}", encoding="utf-8")
    guard = exact.guard
    monkeypatch.setattr(exact, "guard", lambda path: ledger if str(path).endswith("r9_stage2a_audit_use.json") else guard(path))
    with pytest.raises(ValueError, match="already consumed"):
        fit.main(["--source", "synthetic=missing", "--audit-once", "--out", str(tmp_path/"out.json"),
                  "--md", str(tmp_path/"out.md")])
    with pytest.raises(SystemExit):
        fit.main(["--source", "synthetic=missing", "--max-worlds", "8", "--audit-once",
                  "--out", str(tmp_path/"out.json"), "--md", str(tmp_path/"out.md")])


def test_fallback_gate_does_not_impose_refit_self_superiority():
    args = SimpleNamespace(n_quad=16, bootstrap=20, seed=73)
    seqs = [sequence("w"+str(i)) for i in range(4)]
    ev = fit.evaluate(fit.Candidate("F1", initial()), seqs, args)
    ev["paired_vs"] = {"M0": {"nll": {"status": "ok", "ci95": [-2., -1.]},
                                "early_mass": {"status": "ok", "ci95": [-.2, -.1]}},
                        "M0-refit": {"nll": {"status": "ok", "ci95": [0., 0.]}}}
    results = {kind: {"audit": copy.deepcopy(ev), "fit": {"selection_status": "ok"}}
               for kind in ("M0", "M0-refit", "Soft-hazard", "F1")}
    fallback = fit.gates("Soft-hazard", results)
    hybrid = fit.gates("F1", results)
    assert fallback["gates"]["nll_UCB_below_zero_vs_M0"]
    assert "nll_UCB_below_zero_vs_M0-refit" not in fallback["gates"]
    assert not hybrid["gates"]["nll_UCB_below_zero_vs_M0-refit"]


def test_stage2a_cli_fit_only_smoke_preserves_global_audit_and_cache(tmp_path, monkeypatch):
    import numpy as np
    entries = []
    for i in range(10):
        path = tmp_path/("ep"+str(i))
        path.mkdir()
        n = 9
        kin = np.zeros((n, 3, 7), dtype=np.float32)
        kin[..., 0] = np.array([-2., 30., -30.])+np.arange(n)[:, None]*2
        kin[:, 0, 3], kin[:, 1:, 2] = 1., 1.
        kin[..., 4], kin[..., 6] = 4., 5.
        kin[7:, 0, 3], kin[7:, 0, 2] = 0., 1.
        arrays = dict(veh_kin=kin, veh_mask=np.ones((n, 3), dtype=bool),
                      vid=np.tile(np.arange(3), (n, 1)), is_cav=np.zeros((n, 3), dtype=bool),
                      time_s=np.arange(n)*.5)
        for name, value in arrays.items():
            np.save(path/(name+".npy"), value)
        meta = dict(world="w"+str(i), episode="ep"+str(i), seed=963000+i, warmup_s=2.)
        (path/"episode.json").write_text(json.dumps(meta), encoding="utf-8")
        entries.append(dict(dir=path.name, seed=meta["seed"]))
    (tmp_path/"C4_EPISODES_MANIFEST.json").write_text(json.dumps(dict(split="TRAIN", episodes=entries)), encoding="utf-8")
    args = ["--source", "synthetic="+str(tmp_path), "--out", str(tmp_path/"fit.json"),
            "--md", str(tmp_path/"fit.md"), "--cache-dir", str(tmp_path/"cache"),
            "--allow-unverified-physics", "--max-worlds", "8", "--inner-folds", "2", "--iters", "1",
            "--n-quad", "8", "--shrinkage", "0", "--bootstrap", "10", "--workers", "1"]
    assert fit.main(args) == 0
    report = json.loads((tmp_path/"fit.json").read_text(encoding="utf-8"))
    assert report["split"]["audit_worlds"] == ["w4", "w9"]
    assert not report["split"]["audit_used"]
    assert set(report["split"]["fit_worlds"]).isdisjoint(report["split"]["audit_worlds"])
    assert len(report["split"]["fit_worlds"]) == 8
    assert report["cohort"]["retained"] == 8
    assert report["models"]["F1"]["fit"]["final_fit"]["status"] == "fitted"
    assert report["models"]["M0"]["audit"]["cars"] == []
    assert all(r["prefix_decisions"] > 0 for r in report["models"]["F1"]["fit_diagnostics_not_audit"]["cars"])
    # Same safe cache and same deterministic fit when using separate CPU workers.
    args[-1] = "2"
    assert fit.main(args) == 0
    repeated = json.loads((tmp_path/"fit.json").read_text(encoding="utf-8"))
    assert repeated["cohort"] == report["cohort"]
    assert repeated["models"]["F1"]["parameters"] == report["models"]["F1"]["parameters"]
    # An actual one-time synthetic audit, isolated from the real TRAIN ledger.
    ledger = tmp_path/"audit_use.json"
    guard = exact.guard
    monkeypatch.setattr(exact, "guard", lambda path: ledger if str(path).endswith("r9_stage2a_audit_use.json") else guard(path))
    physics = tmp_path/"physics.json"
    physics.write_text(json.dumps(exact.load_frozen().metadata["physics"]), encoding="utf-8")
    audit_args = list(args)
    index = audit_args.index("--max-worlds")
    del audit_args[index:index+2]
    audit_args[-1] = "1"
    audit_args.extend(["--physics", str(physics), "--audit-once"])
    assert fit.main(audit_args) == 0
    audited = json.loads((tmp_path/"fit.json").read_text(encoding="utf-8"))
    assert audited["split"]["audit_used"] and ledger.exists()
    assert audited["cohort"]["retained"] == 10
    assert len(audited["models"]["F1"]["audit"]["cars"]) == 2
    assert audited["frozen_tensors_preserved"]
    with pytest.raises(ValueError, match="already consumed"):
        fit.main(audit_args)


def test_padded_compaction_preserves_present_replay_and_exact_masses():
    frozen = exact.load_frozen()
    k, n = 20, 7
    kin = torch.zeros(n, k, 7)
    kin[:, :3, 0] = torch.tensor([-2., 20., -20.])+torch.arange(n).unsqueeze(1)*2
    kin[:, 0, 3], kin[:, 1:3, 2] = 1., 1.
    kin[:, :3, 4], kin[:, :3, 6] = 4., 5.
    mask = torch.zeros(n, k, dtype=torch.bool)
    mask[:, :3] = True
    cav = torch.zeros_like(mask)
    vid = torch.arange(k).expand(n, -1)
    kwargs = dict(kin=kin, mask=mask, cav=cav, vid=vid, allow_unverified_physics=True)
    fast = heads.prepare_episode(frozen, **kwargs)
    ref = heads.prepare_episode(frozen, **kwargs, vectorized=False)
    for mode in ("original_runtime", "corrected", "seq_tf"):
        eligible = fast[mode]["eligible"][:3]
        a = exact.hazard_distribution(fast[mode]["logits"][:3], fast["weights"], eligible)
        b = exact.hazard_distribution(ref[mode]["logits"][:3], ref["weights"], ref[mode]["eligible"][:3])
        torch.testing.assert_close(a["log_mass"].exp(), b["log_mass"].exp(), atol=1e-6, rtol=0)
        torch.testing.assert_close(a["cdf"], b["cdf"], atol=1e-6, rtol=0)
