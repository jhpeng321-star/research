"""Print compact, explicitly scoped summaries for report assembly."""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[2]/'docs/paper_results/bdrank'


def rows(name):
    with (OUT/(name+'.csv')).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    for name, fields in (
        ('cost_by_status', ('split', 'horizon_s', 'status', 'episodes', 'mean_J', 'mean_source', 'mean_onset_s', 'mean_depth_mps')),
        ('cost_standardized_contrasts', ('split', 'horizon_s', 'supported_cell_arm_groups', 'supported_episodes', 'standardized_event_minus_noevent_J')),
        ('controller_concordance', ('split', 'horizon_s', 'signal', 'target', 'concordant', 'discordant', 'predicted_only_ties', 'real_only_ties', 'both_ties', 'concordance', 'tie_adjusted_concordance')),
        ('controller_decision_summary', ('split', 'horizon_s', 'rule', 'metric', 'mean', 'tie_worlds')),
        ('option_concordance', ('model', 'horizon_s', 'signal', 'target', 'concordant', 'discordant', 'predicted_only_ties', 'concordance')),
        ('option_decision_summary', ('model', 'horizon_s', 'metric', 'mean')),
    ):
        print('\n'+name)
        for r in rows(name):
            if r.get('level', 'pooled') != 'pooled' or r.get('region', 'upstream') != 'upstream':
                continue
            if name == 'controller_concordance' and r['signal'] not in ('occurrence', 'onset_rmst', 'depth_expected', 'mean_delay'):
                continue
            if name == 'controller_decision_summary' and r['metric'] not in ('saving', 'regret'):
                continue
            if name == 'option_concordance' and r['signal'] != 'mean_delay':
                continue
            print(' | '.join(r[k] for k in fields))


if __name__ == '__main__':
    main()
