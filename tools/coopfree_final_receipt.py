"""Read-only integrity/accounting audit of the published final continuation.

No remote access, physics dispatch, source mutation, or scenario/truth-file reads.
Run: python -B -m tools.coopfree_final_receipt
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean


PUBLIC = Path(__file__).resolve().parent/'coopfree/publications/owner1511'


def audit(root=PUBLIC):
    def read(name):
        return json.loads((root/name).read_text(encoding='utf-8'))

    manifest = read('MANIFEST.json')
    for name, expected in manifest.items():
        assert re.fullmatch(r'(?:[A-Z][A-Z0-9_]*\.json|policy_hold_u\d{3}\.pt)', name), name
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    complete, stop = read('CONTINUE_COMPLETE.json'), read('CONTINUE_BUDGET_STOP.json')
    assert complete['last_update'] == 9 and stop['before_update'] == 10
    assert complete['charged'] == stop['charged']
    accounting = complete['charged']
    for phase in accounting.values():
        assert phase['unclosed'] == 0
        assert phase['charged'] == phase['actual_closed'] <= phase['cap']
    total = sum(v['charged'] for v in accounting.values())
    assert total <= 6000000
    updates = []
    for update in range(1, complete['last_update']+1):
        name = f'policy_hold_u{update:03}.pt'
        row = read(f'HOLD_UPDATE_{update:03}.json')
        assert row['checkpoint_sha256'] == manifest[name]
        assert row['windows'] >= 150 and row['worlds'] >= 12
        assert row['exact_credit_windows'] == (66 if update == 1 else 39)
        updates.append(dict(update=update, checkpoint_sha256=manifest[name],
            executed=row['executed_change_share'], override=row['safety_override_share'],
            D_G=row['D_G']['mean'], sum_D_i=row['sum_D_i']['mean'],
            C_G=row['C_G']['mean'], positive_C_G_share=row['positive_C_G_share']))
    discoveries = []
    for update in (3, 6, 9):
        discovery = read(f'CONTINUE_DISCOVERY_{update:03}.json')
        assert 'TRAIN explored proposals' in discovery['limitation']
        groups = discovery['events']
        assert len(groups) == 39
        counts = Counter(e['behaviour'] for g in groups for e in g['events'])
        credits = [g['credit']['600'] for g in groups]
        cluster = discovery['clustering']
        discoveries.append(dict(update=update, scope=discovery['limitation'],
            groups=len(groups), observational_counts=dict(counts),
            D_G=mean(g['D_G'] for g in credits),
            sum_D_i=mean(g['sum_D_i'] for g in credits),
            C_G=mean(g['C_G'] for g in credits),
            positive_C_G_share=mean(g['C_G'] > 0 for g in credits),
            cluster_k=cluster['k'], silhouette=cluster['silhouette'],
            cluster_sizes=dict(Counter(cluster['labels']))))
    return dict(verified_artifacts=len(manifest), sim_s_this_audit=0,
        total_closed=total, remaining_total=6000000-total,
        remaining_train=accounting['train']['cap']-accounting['train']['charged'],
        accounting=accounting, updates=updates, discoveries=discoveries,
        production_complete=True, last_update=complete['last_update'])


if __name__ == '__main__':
    print(json.dumps(audit(), sort_keys=True, indent=2))
