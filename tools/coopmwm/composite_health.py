"""MAIN09:57 SHA-bound safety release and paired composite guard; all other pauses stay active."""
from functools import lru_cache
from pathlib import Path
from .storage import read,write
from .protocol import sha

AUTHORITY='MAIN2026-11-03 09:57'


def rule(root):
    root=Path(root);p=root/'SAFETY_RULE_0957.json'
    if not p.exists():return None
    v=read(p)
    if not (v['authority']==AUTHORITY and v['TTC_threshold']==1.5 and v['bootstrap']==10000
            and v['bootstrap_seed']==20261101 and v['alpha']==.05
            and v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()):
        raise ValueError('unregistered composite safety rule')
    return v


@lru_cache(maxsize=16)
def _released(root,stamps):
    root=Path(root);v=read(root/'SAFETY_RELEASE_0957.json')
    if rule(root) is None:return False
    if not (v.get('authority')==AUTHORITY and v.get('released') is True
            and v.get('original_pause_sha256')==sha(root/'PAUSE_SAFETY.json')
            and v.get('diagnosis_sha256')==sha(root/'SAFETY_DIAGNOSIS_u6.json')
            and v.get('rule_sha256')==sha(root/'SAFETY_RULE_0957.json')
            and v.get('amendment_sha256','UNREGISTERED') in (root/'AUTHORITY.md').read_text()):return False
    d=read(root/'SAFETY_DIAGNOSIS_u6.json')
    from .safety_diagnosis import comparison
    return (not any(d['missing'].values()) and comparison(d['paired_world_rows']['B'])['release']
            and comparison(d['paired_world_rows']['R'])['release'])


LEARNING_AUTHORITY='MAIN2026-11-04 09:00'
LEARNING_PAUSE='PAUSE_V2V_LEARNING_MECHANICS.json'
LEARNING_RELEASE='LEARNING_RELEASE_0900.json'
LEARNING_DIAGNOSIS='LEARNING_DIAGNOSIS_0900.json'


@lru_cache(maxsize=16)
def _learning_released(root,stamps):
    """MAIN 09:00: SHA-bound diagnosis + release of the receipt-only learning-mechanics pause
    (same pattern as the 09:57 safety release). Only this exact pause file is released; any later
    learning pause is written under a new name and stays active."""
    root=Path(root);v=read(root/LEARNING_RELEASE)
    return bool(v.get('authority')==LEARNING_AUTHORITY and v.get('released') is True
        and v.get('pause')==LEARNING_PAUSE
        and v.get('original_pause_sha256')==sha(root/LEARNING_PAUSE)
        and v.get('diagnosis_sha256')==sha(root/LEARNING_DIAGNOSIS)
        and v.get('amendment_sha256','UNREGISTERED') in (root/'AUTHORITY.md').read_text())


def released(root,pause):
    root=Path(root);pause=Path(pause)
    if pause.name==LEARNING_PAUSE:
        names=(LEARNING_RELEASE,LEARNING_DIAGNOSIS,LEARNING_PAUSE,'AUTHORITY.md')
        try:
            stamps=tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in (root/n for n in names))
        except FileNotFoundError:return False
        return _learning_released(str(root),stamps)
    if pause.name!='PAUSE_SAFETY.json':return False
    names=('SAFETY_RELEASE_0957.json','SAFETY_RULE_0957.json','SAFETY_DIAGNOSIS_u6.json','PAUSE_SAFETY.json','AUTHORITY.md')
    try:
        stamps=tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in (root/n for n in names))
    except FileNotFoundError:return False
    return bool(_released(str(root),stamps))


def braking(root,arm,u,episodes,a_records,pause):
    root=Path(root)
    if rule(root) is None:return False
    from .safety_diagnosis import episode,comparison
    a={e['world']['seed']:e for e in a_records};common=[e for e in episodes if e['world']['seed'] in a]
    if len(common)<2:return True  # inherited guard has no estimable paired CI in this case
    cache=root/'safety_diagnosis_0957/ttc_cache';cache.mkdir(parents=True,exist_ok=True)
    rows=[]
    for e in common:
        A=episode(root,a[e['world']['seed']]['id'],True,cache);B=episode(root,e['id'],True,cache)
        if A is None or B is None or A['world']!=B['world']:raise ValueError('composite safety pair incomplete')
        rows.append(dict(seed=e['world']['seed'],world=e['world'],A=A,right=B))
    out=comparison(rows)
    if not out['release']:
        pause(root,f'SAFETY_COMPOSITE_{arm}_u{u:03}',dict(update=u,arm=arm,comparison=out))
    path=root/f'{arm}_SAFETY_COMPOSITE_{u:03}.json'
    if path.exists():
        if read(path)!=out:raise ValueError('completed safety comparison changed')
    else:write(path,out)
    return True
