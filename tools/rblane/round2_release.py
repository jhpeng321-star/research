"""Verify completed owned jobs and publish a byte-bound lane delivery capsule."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from .validate import sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    root=ROOT/'docs/rb2lane'
    out=root/'DELIVERY.json'
    if args.verify:
        report=json.loads(out.read_text())
        expected=dict(report['immutable_artifact_sha256'])
        expected.update({f'docs/rb2lane/{kind}_COMPLETE.json':value['receipt_sha256']
            for kind,value in report['completed_owned_jobs'].items()})
        for name,digest in expected.items():
            assert sha(ROOT/name)==digest, 'working artifact bytes changed: '+name
            archived=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
            assert hashlib.sha256(archived).hexdigest()==digest, 'git artifact bytes changed: '+name
        assert sha(Path(report['shared_result']['path']))==report['shared_result']['sha256']
        print(json.dumps(dict(status='PASS',working_and_committed_artifacts=len(expected),
            source_publication=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            delivery_sha256=sha(out)),indent=2))
        return
    if out.exists():raise ValueError('immutable delivery exists')
    kinds=('smoke_v1','native300_a_v1','native300_b_v1','arms60_a_v1','arms60_b_v1',
        'runtime_v1','free_canary_v1','free_a_v1','free_b_v1','free_c_v1','free_d_v1','prefix_v1')
    completed={kind:json.loads((root/(kind+'_COMPLETE.json')).read_text()) for kind in kinds}
    assert all(r['complete'] and not r['alive'] for r in completed.values())
    suites=list(ET.parse(root/'CPU_FINAL.xml').getroot().iter('testsuite'))
    tests={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','errors','failures','skipped')}
    assert tests==dict(tests=85,errors=0,failures=0,skipped=0)
    # Physical source was frozen before the free pilot; later work only scores it.
    subprocess.run(['git','diff','--exit-code','a60a6a62','--','src/wms','configs/wm_phys_rb2_lane.toml',
        'tools/rbint/recipe.py','tools/rbint/motion_posterior.py','tools/rblane/runtime_history.py',
        'tools/rblane/round2_free_pilot.py'],cwd=ROOT,check=True)
    files=[root/'fit_v2/RESULTS.json',root/'fit_v2/motivation.csv',root/'fit_v2/lane_segment.csv',
        root/'fit_v2/timing_10s.csv',root/'free_v1/RESULTS.json',root/'free_v1/lane_rate_10s.csv',
        root/'free_v1/BOUNDARY_TARGETS.json',root/'runtime_v1_SMOKE.json',root/'INITIAL_LANE_AUDIT.json',
        root/'prefix_v1_results/COMPLETE.json',root/'CPU_FINAL.xml',
        ROOT/'artifacts/rbfit/lane_raw_c5602102891e/lc2013_population.json']
    population=files[-1]
    assert sha(population)=='b903fa49fe1f3f1a15428ae48de5c8f86863aa5c5821dbb0acc8275c77d4483f'
    fit=json.loads((root/'fit_v2/RESULTS.json').read_text())
    free=json.loads((root/'free_v1/RESULTS.json').read_text())
    assert fit['status']==free['status']=='COMPLETE'
    assert free['variants']['fitted_lc']['empirical_K2_count_crps']>free['variants']['source_default']['empirical_K2_count_crps']
    shared=Path('E:/agent-work/rb-shared/RB2_LANE_DEVELOPMENT_RESULT.md')
    report=dict(status='DELIVERED',physics_engineering_ready=True,fitted_lc_adopted=False,
        reason='Conditional FP reduction does not offset worse autonomous development lane counts/CRPS',
        recommendation='keep fitted LC OFF pending actual integrated-source development ON/OFF comparison',
        engineering_commits=['a29116e9','6b4bfbd5'],conditional_publication_commit='a60a6a62',
        free_pilot_source='a60a6a62759ee1312c1d453df2fa7941b517880e',prefix_audit_source='b5e53b3e8762a077522832b0c28f9630cfd37fff',
        physical_code_unchanged_since_freeze=True,core_cpu_tests=tests,additional_guard_scorer_ramp_cpu_tests_passed=5,
        immutable_artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        shared_result=dict(path=str(shared),sha256=sha(shared)),
        completed_owned_jobs={kind:dict(complete=True,alive=False,receipt_sha256=sha(root/(kind+'_COMPLETE.json')))
            for kind in kinds},fit_worlds=list(range(969400,969432)),
        new_autonomous_comparison_worlds=list(range(969420,969432)),reserved_selection=False,wmdata2_access=False,
        neural_training=False,sumo_execution=False,gpu_execution=False,full_distributional_qualification=False)
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],fitted_lc_adopted=False,completed_owned_jobs=len(kinds),
        core_cpu_tests=tests,additional_cpu_tests=5,delivery_sha256=sha(out)),indent=2))


if __name__=='__main__':main()
