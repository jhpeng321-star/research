"""Validate artifact lineage and byte hashes for the integrator (no inference)."""
import argparse
import ast
import hashlib
import json
import io
from pathlib import Path
import subprocess
import tarfile
import tomllib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    def load(path):return json.loads((root/path).read_text())
    def raw_digest(path):return hashlib.sha256((root/path).read_bytes()).hexdigest()
    def digest(path):return hashlib.sha256((root/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    artifact_path = 'artifacts/rb2lphys/learned_v1.json'
    artifact = load(artifact_path)
    original = load('docs/rb2lphys/departure_fit/learned_v1.json')
    support = load('docs/rb2lphys/departure_support_audit/SEARCH.json')
    assert artifact == original
    assert artifact['departure_parameters'] == support['artifact']['departure_parameters']
    assert support['fit_rows'] == 1068 and support['dev_rows'] == 2059
    assert all(p['observable_time_support_s'] == [240.,600.] for p in support['provenance'])
    assert artifact['fit_worlds'] == list(range(969400,969432))
    assert artifact['selection_worlds'] == list(range(969420,969432))
    assert digest('artifacts/rbfit/shrinkage_v1/idm_population.json') == artifact['frozen_idm_population_sha256']
    baseline = tomllib.loads((root/'configs/wm_phys.toml').read_text())
    lp = tomllib.loads((root/'configs/wm_phys_lp.toml').read_text())
    assert lp['physics'].pop('physics_params') == 'learned_v1'
    assert lp['physics'].pop('physics_params_file') == artifact_path
    assert lp == baseline
    paths = [artifact_path,'configs/wm_phys_lp.toml','src/wms/physics/learned_parameters.py',
             'src/wms/model/sumo_insertion.py','src/wms/physics/config.py',
             'tools/rbint/recipe.py','tools/rbint/evaluate.py']
    # git archive on Windows can export CRLF even when the blob is LF. Bind the
    # actual stage bytes and verify newline-only equivalence, rather than calling
    # a local source hash the Linux stage's hash. New artifact attrs pin LF.
    stage_paths = [p for p in paths if p != 'configs/wm_phys_lp.toml']
    archive = subprocess.check_output(['git','archive','--format=tar','06eaf9b3',*stage_paths],cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        stage_bytes = {p:tar.extractfile(p).read() for p in stage_paths}
    assert json.loads(stage_bytes[artifact_path]) == artifact
    assert all(data.replace(b'\r\n',b'\n') == (root/path).read_bytes().replace(b'\r\n',b'\n') for path,data in stage_bytes.items())
    canary = load('docs/rb2lphys/canary_lp/task/IDENTITY.json')
    for path,data in stage_bytes.items():
        hashes = [sha for name,sha in canary['sources'].items() if name.endswith('/'+path)]
        if path == 'tools/rbint/evaluate.py':
            assert not hashes  # This optional generic entrypoint was not used.
        else:
            assert hashes == [hashlib.sha256(data).hexdigest()], path
    for path in (root/'tools/rb2lphys').glob('*.py'):ast.parse(path.read_text())
    result = dict(verified=True,fit_worlds=artifact['fit_worlds'],selection_worlds=artifact['selection_worlds'],
        actual_conditional_check_interval_s=[300,600],requested_conditional_upper_bound_s=900,
        sha256={path:digest(path) for path in paths},
        worktree_raw_sha256={path:raw_digest(path) for path in paths},
        evaluated_archive_sha256={path:hashlib.sha256(data).hexdigest() for path,data in stage_bytes.items()},
        evaluated_archive_commit='06eaf9b3b58f9e137a0f0ee6fc73c3f8e94d2dd1',
        evaluated_values_and_rule_source_equal_modulo_newlines=True,
        hash_convention='sha256 is LF-normalized repository content; evaluated_archive_sha256 is actual historical Windows-git-archive staging (CRLF); raw worktree hashes also recorded',
        WMDATA2_fitting_or_selection=False,reserved8_fitting_or_selection=False,
        inherited_data=['frozen motion encoder/checkpoint; no retraining',
            'rbfit/shrinkage_v1 IDM support/posteriors/shrinkage; FIT969400..969431',
            'rbunc observed-arrival fit; causal own-start conditioning unchanged',
            'canonical rbeval observable reference bundle; only development rows enter scores'],
        correction='Original conditional artifact records requested900s upper bound; actual converted-view support is600s. No parameter value changed.')
    assert args.out.parent.is_dir()
    with args.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
