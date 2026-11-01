import json
from pathlib import Path
import pytest
from tools.rbint.finish_saved_round import reusable_component, saved_snapshot


def test_successful_partial_round_component_is_reused_even_if_ensemble_grew(tmp_path):
    artifact = tmp_path/'completed.json'
    artifact.write_text('{}')
    (tmp_path/'compare_A2_EXIT.json').write_text(json.dumps(dict(returncode=0)))
    assert reusable_component(tmp_path,'compare_A2',artifact,'old-report',False) == str(artifact)


def test_failed_component_cannot_reuse_stale_different_ensemble_report(tmp_path):
    (tmp_path/'compare_FULL-v4_EXIT.json').write_text(json.dumps(dict(returncode=1)))
    assert reusable_component(tmp_path,'compare_FULL-v4',tmp_path/'absent','old-report',False) is None
    assert reusable_component(tmp_path,'compare_FULL-v4',tmp_path/'absent','old-report',True) == 'old-report'


def test_success_marker_alone_cannot_certify_missing_artifact(tmp_path):
    (tmp_path/'compare_A2_EXIT.json').write_text(json.dumps(dict(returncode=0)))
    with pytest.raises(AssertionError,match='artifact missing'):
        reusable_component(tmp_path,'compare_A2',tmp_path/'absent',None,False)


def test_uncollected_variant_preserves_real_older_snapshot_not_new_round_label(tmp_path):
    path, older = saved_snapshot(tmp_path,'C2',dict(snapshot=str(tmp_path/'older-C2')))
    assert older and path == tmp_path/'older-C2'
    (tmp_path/'snapshot_C2_PROCESS.json').write_text(json.dumps(dict(argv=['-u','--out',str(tmp_path/'new-C2')])))
    (tmp_path/'snapshot_C2_EXIT.json').write_text(json.dumps(dict(returncode=0)))
    path, older = saved_snapshot(tmp_path,'C2',dict(snapshot=str(tmp_path/'older-C2')))
    assert not older and path == tmp_path/'new-C2'
