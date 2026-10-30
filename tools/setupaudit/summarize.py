"""Print concise statistics from explicitly named read-only audit receipts."""
import argparse
import json
from pathlib import Path
import statistics


def aggregate(document):
    result = {}
    for actor in dict.fromkeys(r['actor'] for r in document['rows']):
        rows = [r for r in document['rows'] if r['actor'] == actor]
        def mean(values):
            return statistics.mean(values)
        def terms(section):
            return {key: mean(sum(r[section]['decomposition' if section == 'dense'
                    else 'accrued_decomposition'][key]) for r in rows)
                    for key in ('D_all', 'D_G', 'C_G', 'D_i')}
        item = dict(n_starts=len(rows), J_dense=mean(r['dense']['J'] for r in rows),
                    dense_credit=mean(r['dense']['dense_credit'] for r in rows),
                    source_wait_dense=mean(r['dense']['source'] for r in rows),
                    dense_terms=terms('dense'))
        if all('long' in r for r in rows):
            item.update(J300=mean(r['long']['J'] for r in rows), long_terms=terms('long'),
                long_credit=mean(r['long']['gated300_long_credit'] for r in rows),
                gate_decisions=[dict(start=r['start'],
                    eligible=[i for i, c in enumerate(r['dense']['coefficients']) if c != 0],
                    on=r['long']['gated300_active_rows'],
                    median=r['long']['median'], q10=r['long']['q10']) for r in rows])
        n = sum(r['action_delta_vs_ctrl0']['n'] for r in rows)
        item['action_ticks'] = n
        item['reachable_share'] = sum(r['within_residual_envelope_share'] *
            r['action_delta_vs_ctrl0']['n'] for r in rows) / n
        item['positive_share'] = sum(r['positive_delta_share'] *
            r['action_delta_vs_ctrl0']['n'] for r in rows) / n
        item['bridge_lane_changes'] = sum(round(r['bridge_lane_change_share'] *
            r['action_delta_vs_ctrl0']['n']) for r in rows)
        item['delta_min'] = min(r['action_delta_vs_ctrl0']['q'][0] for r in rows)
        item['delta_max'] = max(r['action_delta_vs_ctrl0']['q'][-1] for r in rows)
        result[actor] = item
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--aggregate-only', action='store_true')
    a = p.parse_args()
    for name in a.inputs:
        doc = json.loads(Path(name).read_text())
        print(name, 'PID', doc['pid'], 'mix', doc['learned_mix'])
        for row in ([] if a.aggregate_only else doc['rows']):
            d = row['dense']
            decomp = d['decomposition']
            print(row['actor'], row['start'], 'H', row['dense_horizon'],
                  'J_dense', d['J'], 'dense', d['dense_credit'],
                  'Dall', decomp['D_all'], 'DG', decomp['D_G'], 'CG', decomp['C_G'],
                  'sumDi', sum(decomp['D_i']), 'native90', statistics.mean(d['native_J_futures']))
            if 'long' in row:
                long = row['long']
                print('  J300', long['J'], 'long', long['gated300_long_credit'],
                      'gates', long['gated300_active_rows'],
                      'D300', long['accrued_decomposition'])
            print('  delta', row['action_delta_vs_ctrl0'], 'reachable',
                  row['within_residual_envelope_share'], 'positive', row['positive_delta_share'],
                  'action', row['requested_action'], 'lane', row['bridge_lane_change_share'])
        by_start = {}
        for row in doc['rows']:
            by_start.setdefault(row['start'], {})[row['actor']] = row
        for start, rows in by_start.items():
            if 'CTRL0' not in rows:
                continue
            ctrl = rows['CTRL0']
            for actor, row in rows.items():
                if actor == 'CTRL0':
                    continue
                assert row['dense']['native_J_futures'] == ctrl['dense']['native_J_futures'], (
                    'native background differs', name, start, actor)
                if 'long' not in row:
                    continue
                base = row['dense']['row_kind'].index(1)
                assert row['long']['row_J'][base] == ctrl['long']['row_J'][base], (
                    '300 s native background differs', name, start, actor)
                j90 = row['dense']['J'] - ctrl['dense']['J']
                j300 = row['long']['J'] - ctrl['long']['J']
                credit = (row['dense']['dense_credit'] + row['long']['gated300_long_credit']
                          - ctrl['dense']['dense_credit'] - ctrl['long']['gated300_long_credit'])
                print('PAIRED', start, actor, 'dJ90', j90, 'sigma90', j90 / 25.38,
                      'dJ300', j300, 'sigma300', j300 / 43.91, 'objective_gain', credit)
        print('AGGREGATE', json.dumps(aggregate(doc), indent=2))


if __name__ == '__main__':
    main()
