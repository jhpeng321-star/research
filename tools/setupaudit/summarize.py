"""Print concise statistics from explicitly named read-only audit receipts."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    a = p.parse_args()
    for name in a.inputs:
        doc = json.loads(Path(name).read_text())
        print(name, 'PID', doc['pid'], 'mix', doc['learned_mix'])
        for row in doc['rows']:
            d = row['dense']
            decomp = d['decomposition']
            print(row['actor'], row['start'], 'H', row['dense_horizon'],
                  'J90', d['J'], 'dense', d['dense_credit'],
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
                j90 = row['dense']['J'] - ctrl['dense']['J']
                j300 = row['long']['J'] - ctrl['long']['J']
                credit = (row['dense']['dense_credit'] + row['long']['gated300_long_credit']
                          - ctrl['dense']['dense_credit'] - ctrl['long']['gated300_long_credit'])
                print('PAIRED', start, actor, 'dJ90', j90, 'sigma90', j90 / 25.38,
                      'dJ300', j300, 'sigma300', j300 / 43.91, 'objective_gain', credit)


if __name__ == '__main__':
    main()
