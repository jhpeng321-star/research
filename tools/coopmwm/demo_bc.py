"""c4/learnfix verified-seed demonstration BC (MAIN 2026-11-08 22:45; owner: "running in the preverified seeds and
asking it to perform like the plan").

Extraction is DEMOX's (c4/demox: tools/coopmwm/demo_extract.py); this module is the training side. Design (MAIN):
for each ADMITTED (WM mut_* plan, onset bin) pair of the guidance doc
and each verified activation of that plan in an admitted bin (privileged onset bin, ops/sit_reduce output), replay
the recorded observable tape second by second through the SAME training featurizer (track2_backend.EpisodePolicy
behind the worker's controller wrapper, episode_components) and keep the trigger record at the onset second:
  * act rows: seeds where the plan beat the incumbent (weight = max(0, inc - pick) / max over seeds), the acting
    member's row labelled with the slot offering that plan's template;
  * none rows: seeds where the plan HURT (pick - inc > 0), from the same seed's incumbent and u0 tapes at the same
    onset second, the plan's offered slot labelled 'none' (weight = harm / max harm).
Rows whose slate does not offer the plan are skipped (counted). Verification seeds must be disjoint from the
held-out panel range (map_scenarios.HELDOUT_RANGE) - asserted.

Training: the fixed demo set is added to B's EV-BC every update as its own record kind ('demo'); fullpol_learn._mass
gives each kind equal mass (balanced against the online targets); act and none demo totals are balanced; the same
KL cap applies. R never gets demos. Featurizer parity with training observations: tests + ops/demo_extract.py parity.
"""
import hashlib
import json
from pathlib import Path

RULE = 'c4/learnfix demo-bc v1 (MAIN 2026-11-08 22:45)'
SCHEMA = 'coopmwm.demo_bc.v1'


def assert_seeds_disjoint(seeds):
    from .map_scenarios import HELDOUT_RANGE
    lo, hi = HELDOUT_RANGE
    bad = sorted(int(s) for s in seeds if lo <= int(s) <= hi)
    if bad:
        raise ValueError(f'demo seeds overlap the held-out panel {HELDOUT_RANGE}: {bad}')
    return True


def plan_slot(trigger, family, members=None):
    """(row index, slot) of the first row (members first, in order) whose table offers ``family``'s template."""
    from .fullpol_learn import option_labels
    order = list(range(len(trigger['context']['members'])))
    if members:
        order = sorted(order, key=lambda i: (trigger['context']['members'][i] not in members,
                                             members.index(trigger['context']['members'][i])
                                             if trigger['context']['members'][i] in members else 0))
    for i in order:
        for j, entry in enumerate(trigger['tables'][i]):
            if j and trigger['mask'][i][j] and family in option_labels(entry):
                return i, j
    return None


def demo_records(items):
    """Demo items -> EV-BC records (kind 'demo'); act and none totals balanced; weight <= 1 per row."""
    import torch
    from .track2_backend import _batch
    from .trigger_policy import TriggerBatch
    act = sum(x['weight'] for x in items if x['kind'] == 'act')
    none = sum(x['weight'] for x in items if x['kind'] == 'none')
    scale = dict(act=1., none=(min(1., act/none) if none else 0.))
    if act and none and none < act:
        scale = dict(act=none/act, none=1.)
    records = []
    for x in items:
        w = x['weight']*scale[x['kind']]
        if w <= 0:
            continue
        b = _batch(x['trigger'])
        n, width = b.mask.shape
        actions = torch.zeros(n, dtype=torch.long)
        same = torch.tensor([[j == 0 for j in range(width)] for _ in range(n)])
        row_w = torch.zeros(n)
        i = x['row']
        actions[i] = x['slot'] if x['kind'] == 'act' else 0
        same[i] = torch.tensor([j == (x['slot'] if x['kind'] == 'act' else 0) for j in range(width)])
        row_w[i] = w
        records.append(dict(batch=TriggerBatch(b.context, b.tables, b.descriptors, b.mask, actions, b.logp, b.values),
                            weight=float(w), equivalent=same, row_weight=row_w,
                            meta=dict(kind='demo', episode_id=x['meta']['job'], time_s=x['meta']['t'],
                                      seed=x['meta']['seed'], regime=x['meta']['cell'], family=x['meta']['family'],
                                      label=x['kind'])))
    return records


def load(path, sha256=None):
    raw = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('demo set sha256 differs from its registration')
    doc = json.loads(raw)
    if doc.get('schema') != SCHEMA or doc.get('rule') != RULE:
        raise ValueError('demo set schema/rule differs')
    assert_seeds_disjoint({x['meta']['seed'] for x in doc['items']})
    return doc


def settings(cfg):
    """{'path', 'sha256'} of the registered demo set (cfg['learnfix_demo']) or None."""
    raw = (cfg or {}).get('learnfix_demo')
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {'path', 'sha256'} or len(str(raw['sha256'])) != 64:
        raise ValueError('learnfix_demo needs path and sha256')
    return dict(raw)
