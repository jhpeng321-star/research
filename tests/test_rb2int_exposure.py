import json
import pytest

from tools.rb2int.audit import ROOT
from tools.rb2int.exposure import audit, sha


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture(root):
    fit = list(range(969400, 969432))
    put(root/'docs/rb2int/EVALUATION_REGISTRATION.json', dict(final_comparison_worlds=list(range(969550, 969600))))
    pop = root/'artifacts/rbfit/shrinkage_v1/idm_population.json'
    put(pop, {})
    shrink = root/'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json'
    put(shrink, dict(fit_worlds=fit, idm_population_sha256=sha(pop)))
    arrival = root/'docs/paper_results/rbunc/observed_arrivals.json'
    put(arrival, {})
    cal = root/'docs/paper_results/rb2cal/v1/fit/calibration.json'
    put(cal, dict(fit_worlds=fit, idm_population_sha256=sha(pop), shrinkage_sha256=sha(shrink),
                  arrivals_sha256=sha(arrival), learned_corrections=False))
    lp = root/'artifacts/rb2lphys/learned_v1.json'
    put(lp, dict(fit_worlds=fit, selection_worlds=list(range(969420, 969432)), frozen_idm_population_sha256=sha(pop)))
    lane = root/'artifacts/rbfit/lane_raw_c5602102891e/lc2013_population.json'
    put(lane, dict(fit_worlds=fit))
    put(lane.parent/'PACKAGE.json', dict(artifact_sha256={'lc2013_population.json': sha(lane)},
                                       saved_driver_id_lookup=False, population_refit=False))
    for name in ('wm_phys_v5.toml', 'wm_phys_v5_lp.toml'):
        dest = root/'configs'/name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text((ROOT/'configs'/name).read_text(), encoding='utf-8')
    for name in ('configs/execparity_cav_idm_v1.json', 'configs/native_cav_head_v5.json',
                 'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz'):
        put(root/name, {})
    return lp, cal


def test_no_declared_fit_overlap_is_distinct_from_final_execution_authority(tmp_path):
    fixture(tmp_path)
    result = audit(tmp_path)
    assert result['fit_separation_passed'] is True
    assert result['checkpoint_verified'] is False
    assert result['final_inference_authorized'] is False


def test_even_one_additional_final_world_in_physical_fit_is_rejected(tmp_path):
    lp, _ = fixture(tmp_path)
    obj = json.loads(lp.read_text())
    obj['fit_worlds'].append(969550)
    put(lp, obj)
    with pytest.raises(ValueError, match='exposure overlaps'):
        audit(tmp_path)


def test_inherited_artifact_hash_mismatch_cannot_clear_exposure(tmp_path):
    _, cal = fixture(tmp_path)
    obj = json.loads(cal.read_text())
    obj['shrinkage_sha256'] = 'wrong'
    put(cal, obj)
    with pytest.raises(ValueError, match='inherited artifact lineage'):
        audit(tmp_path)


def test_prohibited_audit_root_is_rejected_before_any_io(tmp_path):
    with pytest.raises(ValueError, match='prohibited audit path'):
        audit(tmp_path/'sealed')
