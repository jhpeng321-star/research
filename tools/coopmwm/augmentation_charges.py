"""Receipt-directory-qualified IDs for native all8 CRNs and later-harm waves.

Native verification_id deliberately stays constant across CRN replicas of the
same proposal. It is not a globally unique single-ledger attempt identity.
"""
from pathlib import Path

from .protocol import digest

SCHEME = 'native_directory_v2'


def namespace(branches_root, snapshot_sha256, window):
    from .rerun_cpus import guard_path
    root = Path(guard_path(Path(branches_root).resolve())).as_posix()
    return digest(dict(branches_root=root, snapshot_sha256=snapshot_sha256, window=window))[:20]


def job_id(spec):
    prefix = namespace(spec['t2c_artifact_root'], spec['snapshot_sha256'], spec['window'])
    return f'aug_{prefix}_{spec["verification_id"]}_{spec["branch"]}'


def receipt_ids(path, receipt):
    prefix = namespace(Path(path).parent.parent/'branches', receipt['snapshot_sha256'], receipt['window'])
    return [f'aug_{prefix}_{receipt["verification_id"]}_{b["branch"]}' for b in receipt['branches']]


def recorded_binding(root, path, receipt):
    """Bind actual charges, or the explicitly preserved initial completed wave."""
    from .launch_training import read
    from .exact_grade import verify_sealed
    from .storage import ledger
    root, path = Path(root), Path(path).resolve()
    charged = {r['id']:r for r in ledger(root)}
    ids = receipt_ids(path, receipt)
    if all(i in charged for i in ids):
        return dict(charge_scheme=SCHEME, charge_ids=ids)
    resume = read(root/'AUGMENTATION_RESUME.json')
    verify_sealed(resume, 'receipt_sha256')
    legacy = [f'{receipt["verification_id"]}_{b["branch"]}' for b in receipt['branches']]
    match = next((r for r in resume['completed_receipts'] if Path(r['path']).resolve() == path
                  and r['receipt_sha256'] == receipt['receipt_sha256'] and r['charge_ids'] == legacy), None)
    if match is None or not all(i in charged for i in legacy):
        raise ValueError('completed augmented receipt lacks its own namespaced precharge')
    return dict(charge_scheme='legacy_unqualified', charge_ids=legacy,
                adopted_ledger_binding_sha256=resume['receipt_sha256'])
