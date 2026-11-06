"""CPU tests: labels, ranking, gate, keep-speed backend and the run_qual pipeline (fakes only)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.wzqual import gate as G
from tools.wzqual import labels as L
from tools.wzqual import ranking as RK
from tools.wzqual import run_qual as RQ
from tools.wzqual.backends import KeepSpeedBackend, ReplayBackend
from tools.wzqual.traj import load_wzqual_tape, start_state

from .fakes import write_tape

W = dict(map="wz3to2", cell="D3200_p60", demand=3200, penetration=0.6, seed=989201)


def _hold(vid, lane, **p):
    return dict(id=vid, lane=lane, params=dict(dict(release_x=2696.0, amax=1.0, v_w=22.0, hold_s=60, dmax=1.0), **p))


def _write_labels(d: Path, t0=450):
    sid = f"nm2_D3200_p60_s989201_t{t0}"
    b = dict(schema="coopmwm-mined-move-v1", arm="B", seed=1, situation_id=sid, world=W, t0=t0,
             family="closing_pair", variant="pair_g", members=["main.3", "main.4"], member_lanes=[0, 1],
             move=dict(members=[_hold("main.3", 0), _hold("main.4", 1)]),
             advantage_veh_s={"1": 900.0}, adv_lower95={"1": 100.0}, synergy_veh_s={"1": 300.0},
             synergy_lower95={"1": 50.0}, solo_gain_veh_s={"main.3": 200.0, "main.4": 100.0},
             best_solo="main.3", replicate_summary=dict(T=[900.0, 800.0, 700.0], S=[300.0, 250.0, 200.0]))
    b2 = dict(b, seed=2, advantage_veh_s={"2": 800.0})
    negs = []
    for i, (variant, t) in enumerate([("pair_h", -200.0), ("wall_g", 50.0), ("R0", 400.0)]):
        negs.append(dict(schema="newmap2-negative-v1", arm="R" if variant == "R0" else "B", situation_id=sid,
                         world=W, t0=t0, family="R_random" if variant == "R0" else "abreast_wall",
                         variant=variant, members=["main.3", "main.6"], member_lanes=[0, 2],
                         params=dict(v_w=15.0, hold_s=60, dmax=1.5), admitted=False, stage="screen",
                         advantage_veh_s=t, adv_lower95=t - 100, synergy_veh_s=None, synergy_lower95=None,
                         reps=None))
    ck = dict(schema="coopkinds-negative-v1", arm="B", situation_id=f"ck_D3200_p60_s989201_t{t0}", world=W,
              t0=t0, family="lane_balance", variant="B_lane_balance", members=["main.5"], status="screen_fail_LB95T",
              admitted=False, T=-50.0, T_lb95=-300.0, S=None, S_lb95=None, best_solo=None, solo_gain=None,
              r1=None, r2=None, move=dict(members=[dict(id="main.5", role="lane_to", lane=0,
                                                         params=dict(target_lane=2, giveup_x=2400.0, hold_s=60))]))
    (d / "NEWMAP_MOVES_B.jsonl").write_text(json.dumps(b) + "\n" + json.dumps(b2) + "\n")
    (d / "NEWMAP_NEGATIVES.jsonl").write_text("".join(json.dumps(n) + "\n" for n in negs))
    (d / "COOPKINDS_NEGATIVES.jsonl").write_text(json.dumps(ck) + "\n")


def test_labels_parse_all_schemas(tmp_path):
    _write_labels(tmp_path)
    c = {x.variant: x for x in L.load_candidates(tmp_path)}
    assert set(c) == {"pair_g", "pair_h", "wall_g", "R0", "B_lane_balance"}
    g = c["pair_g"]
    assert g.T == 900.0 and g.S == 300.0 and g.admitted and g.reps_T == [900.0, 800.0, 700.0]
    assert g.members[0]["role"] == "hold" and g.spec_source == "row.move"
    h = c["pair_h"]                                      # registered nm2 reconstruction
    assert h.spec_source == "nm2_reconstruction" and h.members[0]["params"]["release_x"] == 2696.0
    assert h.T == -200.0 and h.S is None and not h.admitted
    assert c["B_lane_balance"].members[0]["role"] == "lane_to" and c["B_lane_balance"].T == -50.0
    assert c["pair_g"].moment == "989201:D3200_p60@450"
    mom = L.load_moments([])
    assert mom == {}


def test_pair_t_reconstruction_uses_closing_lane_anchor():
    row = dict(members=["a", "b"], member_lanes=[1, 0], params=dict(v_w=22.0, hold_s=150, dmax=1.0,
                                                                     anchor_release_x=2546.0), family="closing_pair")
    spec = L._nm2_reconstruct(row)
    assert all(m["params"]["anchor"] == "b" for m in spec)


def _rows(pred, true, S=None, moments=None):
    rows = []
    for i, (p, t) in enumerate(zip(pred, true)):
        rows.append(dict(cid=f"c{i}", moment=(moments[i] if moments else f"m{i // 4}"), seed=i // 4,
                         family="f", arm="B", pred_T=p, T_true=t,
                         pred_S=(S[i][0] if S else None), S_true=(S[i][1] if S else None),
                         S_lb_true=(S[i][1] if S else None), admitted=False))
    return rows


def test_ranking_perfect_and_reversed():
    true = list(np.linspace(-1000, 3000, 40))
    perfect = RK.action_response(_rows(true, true, S=[(t, t) for t in true]))
    assert perfect["pooled_spearman_T"] == pytest.approx(1.0)
    assert perfect["within_moment_spearman_T"] == pytest.approx(1.0)
    assert perfect["precision_positive_T"]["precision"] == 1.0
    assert perfect["precision_at_k_S"]["precision"] == 1.0
    assert perfect["regret"]["wm_regret"] == pytest.approx(0.0)
    rev = RK.action_response(_rows([-x for x in true], true))
    assert rev["pooled_spearman_T"] == pytest.approx(-1.0)
    assert RK.spearman([1, 1, 1], [1, 2, 3]) is None
    assert RK.spearman([1, 2, 2, 3], [1, 2, 3, 4]) > 0.9
    assert RK.k_for(100) == 10 and RK.k_for(7) == 3


def test_regret_semantics():
    rows = _rows([10.0, -5.0, 3.0], [100.0, 500.0, -50.0], moments=["m", "m", "m"])
    r = RK.regret(rows)
    assert r["wm_gain"] == 100.0 and r["wm_regret"] == 400.0
    assert r["random_regret"] == pytest.approx(500.0 - np.mean([100.0, 500.0, -50.0]))
    none = RK.regret(_rows([-1.0, -2.0], [100.0, 50.0], moments=["m", "m"]))
    assert none["wm_gain"] == 0.0 and none["acted_frac"] == 0.0


def _passing_metrics():
    dist = dict(n_starts=200, n_starts_not_broken=120, rule_degenerate=False, n_truth_events=60,
                backend_models_entries=True, rate_diff_pooled=0.05,
                rate_diff_by_cell={"D3200_p60": dict(n=40, diff=0.1), "D2400_p20": dict(n=5, diff=0.9)},
                onset_w1_s=20.0, queue_rel_mean_err=0.1, queue_norm_w1=0.3, throughput_rel_err=0.03,
                zone_crps={"120": dict(wm=3.0, keep_speed=5.0), "300": dict(wm=6.0, keep_speed=8.0)})
    act = dict(n_moments_ranked=60, within_moment_spearman_T=0.4, within_moment_spearman_T_ci=[0.1, 0.6],
               pooled_spearman_T=0.5, pooled_spearman_T_ci=[0.3, 0.6],
               precision_positive_T=dict(precision=0.75, base_rate=0.55),
               precision_at_k_S=dict(n=80, k=10, precision=0.8, base_rate=0.58),
               regret=dict(ratio_wm_to_random=0.3, wm_gain_ci=[100.0, 900.0], wm_gain=500.0),
               keep_speed=dict(pooled_spearman_T=0.2))
    return dict(integrity=dict(I1_world_disjoint=True, I2_truth_identity=True, I3_rule_frozen_first=True,
                               I4_backend_contract=True, I5_no_restricted_seeds=True),
                role_coverage=dict(fraction_T_candidates_supported=0.95, supported_roles=["hold", "lane_to", "pulse"]),
                distributional=dist, action_response=act)


def test_gate_verdicts():
    spec, sha = G.load_spec()
    assert len(sha) == 64 and spec["schema"] == "wzqual_gate/1"
    m = _passing_metrics()
    v = G.evaluate(m, spec)
    assert v["verdict"] == "PASS", v
    # per-cell check ignores cells below the minimum sample size (D2400_p20 n=5)
    f = copy.deepcopy(m)
    f["distributional"]["rate_diff_pooled"] = -0.18           # the merge-map failure size
    assert G.evaluate(f, spec)["verdict"] == "FAIL"
    f = copy.deepcopy(m)
    f["action_response"]["precision_positive_T"] = dict(precision=0.41, base_rate=0.365)
    assert G.evaluate(f, spec)["criteria"]["A3_precision_positive_T"]["status"] == "FAIL"
    f = copy.deepcopy(m)
    f["action_response"]["precision_at_k_S"] = dict(n=80, k=10, precision=0.7, base_rate=0.9)
    assert G.evaluate(f, spec)["criteria"]["A4_precision_at_k_S"]["status"] == "PASS"   # lift cap 0.70
    u = copy.deepcopy(m)
    u["distributional"]["rule_degenerate"] = True
    assert G.evaluate(u, spec)["verdict"] == "INCONCLUSIVE"
    u = copy.deepcopy(m)
    u["distributional"]["backend_models_entries"] = False
    assert G.evaluate(u, spec)["criteria"]["D1_breakdown_rate"]["status"] == "UNIDENTIFIED"
    i = copy.deepcopy(m)
    i["integrity"]["I1_world_disjoint"] = False
    assert G.evaluate(i, spec)["verdict"] == "INVALID"
    c = copy.deepcopy(m)
    c["role_coverage"]["supported_roles"] = ["hold"]
    assert G.evaluate(c, spec)["criteria"]["C0_role_coverage"]["status"] == "FAIL"
    k = copy.deepcopy(m)
    k["action_response"]["keep_speed"]["pooled_spearman_T"] = 0.6
    assert G.evaluate(k, spec)["criteria"]["A6_beats_keep_speed"]["status"] == "FAIL"


def test_keep_speed_backend_contract_and_member_control(tmp_path):
    tr = load_wzqual_tape(write_tape(tmp_path / "t" / "TAPE.npz", t_end=700.0, n_veh=120))
    st = start_state(tr, 400.0)
    vid = st.frame["ids"][len(st.frame["ids"]) // 2]
    arms = {"base": [], "group": [dict(id=vid, role="hold", params=dict(v_w=10.0, hold_s=30, dmax=2.0, amax=1.0,
                                                                        release_x=2696.0))]}
    out = KeepSpeedBackend().rollout(st, arms, horizon_s=60.0, n_samples=2, seed=1)
    assert set(out) == {"base", "group"} and len(out["base"]) == 2
    assert RQ.check_contract(st, out) == []
    b, g = out["base"][0], out["group"][0]
    vb = b.frame(b.step_at(410.0))
    vg = g.frame(g.step_at(410.0))
    i_b, i_g = vb["ids"].index(vid), vg["ids"].index(vid)
    assert vg["v"][i_g] < vb["v"][i_b] - 5.0                  # the held member slowed down
    assert any(n.startswith("__entry_") for n in b.ids)       # persistence inflow
    rep = ReplayBackend({st.world: tr}).rollout(st, {"base": []}, horizon_s=60.0)
    assert RQ.check_contract(st, rep) == []


def test_train_world_seeds_parser(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(dict(train=["w989101m3200p60t300"], val=[dict(seed=989150)], other="989199")))
    assert RQ.train_world_seeds(str(p)) == {989101, 989150, 989199}
    assert RQ.restricted(989600) and RQ.restricted(989100) and not RQ.restricted(989300)


def test_run_qual_end_to_end_keep_speed(tmp_path):
    tapes = tmp_path / "tapes"
    for seed in (989201, 989202):
        write_tape(tapes / f"s{seed}" / "TAPE.npz", seed=seed, t_end=900.0, n_veh=150, slow_after=500.0)
    labels = tmp_path / "labels"
    labels.mkdir()
    _write_labels(labels)
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps(dict(worlds=["w989105m3200p60t300"])))
    out = tmp_path / "out"
    rc = RQ.main(["--out", str(out), "--tapes", str(tapes / "*" / "TAPE.npz"), "--labels-dir", str(labels),
                  "--backend", "keep_speed", "--train-worlds", str(manifest)])
    assert rc == 0
    m = json.loads((out / "WM_QUAL_METRICS.json").read_text())
    v = json.loads((out / "GATE_VERDICT.json").read_text())
    assert m["integrity"]["I1_world_disjoint"] is True
    assert m["distributional"]["n_starts"] == 2 * 3                      # t0 = 300, 420, 540
    assert m["role_coverage"]["n_T"] == 5
    assert v["verdict"] in ("INCONCLUSIVE", "FAIL")                      # too few samples to PASS
    assert (out / "BREAKDOWN_RULE.json").exists()
    rows = json.loads((out / "ACTION_ROWS.json").read_text())["wm"]
    assert {r["J_mode"] for r in rows} == {"road"}
    with pytest.raises(FileExistsError):                                  # never overwrite a run
        RQ.main(["--out", str(out), "--tapes", str(tapes / "*" / "TAPE.npz"), "--labels-dir", str(labels)])
    # overlap with the training worlds is an integrity failure
    manifest.write_text(json.dumps(dict(worlds=["w989201m3200p60t300"])))
    out2 = tmp_path / "out2"
    RQ.main(["--out", str(out2), "--tapes", str(tapes / "*" / "TAPE.npz"), "--labels-dir", str(labels),
             "--skip-action", "--train-worlds", str(manifest)])
    assert json.loads((out2 / "GATE_VERDICT.json").read_text())["verdict"] == "INVALID"


def test_sumo_planner_refuses_restricted_seeds(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sumo"))
    import wmd_plan
    nat = tmp_path / "root" / "runs" / "D3200_p60" / "s989600" / "native"
    nat.mkdir(parents=True)
    (nat / "RESULT.json").write_text(json.dumps(dict(status="ok", job=dict(seed=989600, demand={"main": 3200},
                                                                          penetration=0.6, save_times=[450]))))
    with pytest.raises(SystemExit):
        wmd_plan.plan([str(tmp_path / "root")], 1200.0, tmp_path)
    ok = tmp_path / "root2" / "runs" / "D3200_p60" / "s989300" / "native"
    ok.mkdir(parents=True)
    (ok / "RESULT.json").write_text(json.dumps(dict(status="ok", job=dict(seed=989300, demand={"main": 3200},
                                                                         penetration=0.6, save_times=[450]))))
    jobs = wmd_plan.plan([str(tmp_path / "root2")], 1200.0, tmp_path)
    assert len(jobs) == 1 and jobs[0]["cell"] == "D3200_p60" and jobs[0]["t_end"] == 1200.0


def test_sumo_identity_check(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sumo"))
    import wmd_check
    ref = dict(status="ok", routes_sha256="r", J_road=[1.0, 2.0], J_src=[0.0, 0.0], arrivals=[0, 1],
               snapshots={"t450": dict(cavs=[dict(id="a", edge="up", lane=0, x=1.0, v=2.0, leader_gap=3)],
                                       inserted=5)})
    mine = dict(ref, J_road=[1.0, 2.0, 9.0], J_src=[0.0, 0.0, 0.0], arrivals=[0, 1, 0],
                snapshots={"t450": dict(cavs=[dict(id="a", edge="up", lane=0, x=1.0, v=2.0)], inserted=5)},
                collisions=0, teleports=0, tape_sha256="s")
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    out = tmp_path / "job"
    out.mkdir()
    (out / "RESULT.json").write_text(json.dumps(mine))
    job = dict(job_id="j", seed=989300, cell="D3200_p60", out_dir=str(out), ref_native=str(tmp_path / "ref.json"))
    assert wmd_check.check(job)["identical"] is True
    mine["J_road"][1] = 2.5
    (out / "RESULT.json").write_text(json.dumps(mine))
    assert wmd_check.check(job)["identical"] is False
