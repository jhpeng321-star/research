"""Publish generated pulse exports while preserving the earlier blocked publication."""
from pathlib import Path
import json
import shutil
from core import digest

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'docs/paper_results'


def main():
    staged=OUT/'c1floor_pulse'
    receipt=json.loads((staged/'C1FLOOR_ANALYSIS.json').read_text())
    assert receipt['method']=='SUMO_tiny_perturbation_chaos' and receipt['worlds']==6
    old=json.loads((OUT/'C1FLOOR_ANALYSIS.json').read_text())
    assert old['status']=='BLOCKED_SAVED_STATE_CANARY'
    assert digest(OUT/'C1FLOOR_error_horizon.csv')==old['csv_sha256']
    backup=HERE/'evidence/blocked_publication'
    assert backup.parent.is_dir();backup.mkdir()
    for name in ('C1FLOOR_error_horizon.csv','C1FLOOR_ANALYSIS.json'):
        shutil.copy2(OUT/name,backup/name)
    for name in ('C1FLOOR_error_horizon.csv','C1FLOOR_per_world.csv','C1FLOOR_per_draw.csv','C1FLOOR_ANALYSIS.json'):
        shutil.copy2(staged/name,OUT/name)
        assert digest(staged/name)==digest(OUT/name)
    print('Published pulse CSVs; original blocked export preserved in evidence/blocked_publication')


if __name__=='__main__':main()
