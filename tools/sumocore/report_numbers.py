"""Compact region diagnostics from the completed, hashed score receipt."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics

ROOT=Path(__file__).resolve().parents[2]/'docs/paper_results/sumocore/scores'


def main():
    groups=defaultdict(list)
    with (ROOT/'speed.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['horizon']=='600':groups[r['variant'],r['region']].append(r)
    for (variant,region),rows in groups.items():
        print(json.dumps(dict(variant=variant,region=region,
            crps=statistics.mean(float(r['crps']) for r in rows if r['crps']),
            cell_climatology_crps=statistics.mean(float(r['cell_climatology']) for r in rows if r['cell_climatology']))))


if __name__=='__main__':main()
