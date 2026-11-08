"""c4/learnfix bank growth at update boundaries (owner 2026-11-08 22:49 (4), MAIN): newly admitted WM plans and their
verified-seed demos may join the CLEAN5C5 guidance / demo set ONLY at an update boundary.

Registration ``bank_growth`` = {rule, control}: ``control`` is a JSON file {guidance_path, guidance_sha256 [,
demo_path, demo_sha256]} that operators repoint between updates. At each B/R update start (launch_training.update):
  * B (always the first trained arm of an update) freezes the update's snapshot: both files are sha-checked, copied
    immutably into run_root/bank_growth/ and recorded in bank_growth/U<u>.json (exclusive create);
  * R reads the SAME snapshot (missing -> fail closed), so B and R always use the identical bank in an update;
  * the snapshot's guidance replaces only the guidance-derived config fields (clean5_ev.regime_guidance /
    parity_families, the B = R LCEXEC templates, the clean5c receipt), exactly as clean5b_restart.keys_c derives
    them; the demo set replaces learnfix_demo. CONFIG.json, the policy and every optimizer are never touched.
Each update receipt carries {update, guidance_sha256, demo_sha256, changed_vs_config}. Absent registration: no-op.
"""
from copy import deepcopy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

#: MAIN 2026-11-09 ~05:3x: optional slate override - the FULLPOL free-program generator offers ONLY the bank's verified
#: templates (no structured / random free programs), B = R; applied through the control at an update boundary.
VERIFIED_ONLY = dict(rule='c4/strongguide verified-only slate v1 (MAIN 2026-11-09): free-program classes = [verified]',
                     classes=['verified'], structured=[])
RULE = 'c4/learnfix bank-growth v1 (owner 2026-11-08 22:49 (4)): update-boundary snapshots, B freezes, R reuses'


def settings(cfg):
    raw = (cfg or {}).get('bank_growth')
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {'rule', 'control'} or raw['rule'] != RULE or not raw['control']:
        raise ValueError('bank_growth needs the registered rule and a control path')
    return dict(raw)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy(src, sha, folder, stem):
    if _sha(src) != sha:
        raise ValueError(f'bank growth: {src} sha256 differs from the control file')
    dst = folder/f'{stem}_{sha[:16]}.json'
    if not dst.exists():
        tmp = dst.with_suffix('.tmp')
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    if _sha(dst) != sha:
        raise ValueError('bank growth: immutable copy differs')
    return str(dst)


def snapshot(cfg, arm, update):
    """The update's frozen snapshot (B creates it; R reuses it)."""
    reg = settings(cfg)
    folder = Path(cfg['run_root'])/'bank_growth'
    path = folder/f'U{int(update):03}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    if arm != 'B':
        raise ValueError(f'bank growth: {path} missing (B freezes the update snapshot first; fail closed)')
    folder.mkdir(parents=True, exist_ok=True)
    control = json.loads(Path(reg['control']).read_text(encoding='utf-8'))
    snap = dict(rule=RULE, update=int(update), control=reg['control'], unix=time.time(),
                guidance_sha256=control['guidance_sha256'],
                guidance_copy=_copy(control['guidance_path'], control['guidance_sha256'], folder, 'guidance'))
    if control.get('demo_path'):
        snap.update(demo_sha256=control['demo_sha256'],
                    demo_copy=_copy(control['demo_path'], control['demo_sha256'], folder, 'demo'))
    if control.get('learnfix2_override') is not None:
        from .learnfix import STRONG
        if control['learnfix2_override'] != STRONG:
            raise ValueError('bank growth: learnfix2_override must be exactly learnfix.STRONG')
        snap['learnfix2_override'] = control['learnfix2_override']
    if control.get('slate_override') is not None:
        if control['slate_override'] != VERIFIED_ONLY:
            raise ValueError('bank growth: slate_override must be exactly bank_growth.VERIFIED_ONLY')
        snap['slate_override'] = control['slate_override']
    with open(path, 'x', encoding='utf-8') as fh:            # exclusive: one snapshot per update
        json.dump(snap, fh, indent=1, sort_keys=True)
    return snap


def apply_guidance(native, doc):
    """The guidance-derived config fields from ``doc`` (as keys_c derives them); returns a new config."""
    from . import fullpol_bank, lc_families
    from .clean5b_restart import cells_view, check_p20_guidance, parity_receipt, unique_templates
    out = deepcopy(native)
    cells = tuple(out['newmap']['cell'])
    doc = cells_view(doc, cells)              # MAIN: the UNION bank, training-cell rows only
    check_p20_guidance(doc, cells, wm=True)
    unique = unique_templates(doc)
    out['clean5_ev']['regime_guidance'] = deepcopy(doc)
    out['clean5_ev']['parity_families'] = fullpol_bank.eligible_families(doc)
    out[lc_families.CONFIG_KEY]['generator_args']['templates'] = dict(B=unique, R=json.loads(json.dumps(unique)))
    out['clean5c'] = parity_receipt(doc)
    return out


def update_view(native, arm, update, *, smoke=False):
    """(config for this update, receipt | None). A, smoke and unregistered runs: unchanged."""
    if settings(native) is None or arm == 'A' or smoke:
        return native, None
    snap = snapshot(native, arm, update)
    raw = Path(snap['guidance_copy']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != snap['guidance_sha256']:
        raise ValueError('bank growth: snapshot guidance copy differs')
    doc = json.loads(raw)
    out = apply_guidance(native, doc)
    if snap.get('demo_copy'):
        out['learnfix_demo'] = dict(path=snap['demo_copy'], sha256=snap['demo_sha256'])
    if snap.get('learnfix2_override') is not None:
        from .learnfix import apply_strong
        if out.get('learnfix2') is None:
            raise ValueError('bank growth: learnfix2_override needs a learnfix2-registered run')
        out['learnfix2'] = apply_strong(out['learnfix2'], snap['learnfix2_override'])
    if snap.get('slate_override') is not None:
        from . import lc_families
        args = out[lc_families.CONFIG_KEY]['generator_args']
        args['classes'], args['structured'] = list(VERIFIED_ONLY['classes']), list(VERIFIED_ONLY['structured'])
    changed = (out['clean5_ev']['regime_guidance'] != native['clean5_ev']['regime_guidance']
               or out.get('learnfix_demo') != native.get('learnfix_demo')
               or out.get('learnfix2') != native.get('learnfix2')
               or snap.get('slate_override') is not None)
    return out, dict(rule=RULE, update=int(update), arm=arm, guidance_sha256=snap['guidance_sha256'],
                     guidance_rows_sha256=doc['source']['sha256'], demo_sha256=snap.get('demo_sha256'),
                     changed_vs_config=bool(changed), learnfix2_override=snap.get('learnfix2_override'),
                     slate_override=snap.get('slate_override'),
                     families=sorted({r['family'] for r in doc['rows']}))
