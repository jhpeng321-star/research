import json

from tools.coopbehav.sources import build_source_index


def test_sumocf_adapter_uses_saved_subset_roster_reconstructs_credit_and_is_stable(tmp_path):
    root=tmp_path/"sumocf"; job=root/"jobs"/"probe_01"; job.mkdir(parents=True)
    traces=root/"traces"/"probe_01"; traces.mkdir(parents=True)
    branches=[]
    for i,(members,cost) in enumerate([([],10),(["a"],9),(["b"],10),(["a","b"],8)]):
        path=traces/f"b{i:03d}.jsonl.gz"; path.write_bytes(b"closed-test-file")
        branches.append(dict(members=members,J600=cost,trace=str(path)))
    result=dict(job=dict(seed=969200,t0=400,window=30,checkpoint_sha256="abcdef"),group=["a","b"],
        branches=branches,credit=dict(D_G=2,sum_D_i=1,C_G=1),initial_state_sha256="state")
    (job/"result.json").write_text(json.dumps(result))
    out=tmp_path/"analysis"
    status=build_source_index("sumocf",root,out)
    index=json.loads((out/"COOPBEHAV_INDEX.json").read_text())
    assert status["pairs"]==1 and index["ready"]
    pair=index["pairs"][0]
    assert pair["kind"]=="policy_mean_intervention" and pair["C_G"]==1
    assert pair["factual"].endswith("b003.jsonl.gz") and pair["reference"].endswith("b000.jsonl.gz")
    assert pair["branch"]["horizon_s"]==60 and pair["intervention_window_s"]==30
    first=(out/"COOPBEHAV_INDEX.json").read_bytes()
    build_source_index("sumocf",root,out)
    assert (out/"COOPBEHAV_INDEX.json").read_bytes()==first


def test_audit_adapter_reports_missing_schema_instead_of_guessing_lengths_or_actions(tmp_path):
    job=tmp_path/"audit"/"jobs"/"main_s969650"; job.mkdir(parents=True)
    (job/"result.json").write_text('{"ok":true}')
    out=tmp_path/"analysis"
    status=build_source_index("coopaudit",tmp_path/"audit",out)
    assert status["pairs"]==0 and "lacks length_m" in status["blocked"][0]["reason"]
    assert not json.loads((out/"COOPBEHAV_INDEX.json").read_text())["ready"]
