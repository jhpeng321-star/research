"""A38 demand pick (zero physics), from ops/a38_road2.py calsummary output. Rule (proposed in PROGRESS.md, binding only
once MAIN registers it, BEFORE calibration runs): per demand D in (1000, 1200, 1400) the U0 calibration block (8 TRAIN-type
worlds, p20) gives the mean approach speed v(D) (all logged rows upstream of the drop over [300, 1200)) and safety totals.
  gridlocked(D)  := v(D) < 5.0 m/s  or  collisions > 0  or  teleports > 0
  congested(D)   := v(D) <= 25.0 m/s   (free flow ~ 31.29)
Pick the HIGHEST D that is congested and not gridlocked; none -> exit 3 (chain stops, MAIN decides).
usage: python pick_demand.py CALSUMMARY.json  -> prints a38_D<d>_p20"""
import json, sys

V_GRID, V_CONG = 5.0, 25.0


def pick(doc):
    ok = []
    for name, row in doc['cells'].items():
        v = row.get('mean_approach_speed_mps')
        st = row.get('safety_totals') or {}
        if v is None or row.get('worlds_published') != row.get('worlds'):
            raise SystemExit(f'{name}: calibration incomplete or no vehicle logs')
        grid = v < V_GRID or st.get('collisions', 0) > 0 or st.get('teleports', 0) > 0
        if v <= V_CONG and not grid:
            ok.append(int(name.split('_D')[-1]))
    return f'a38_D{max(ok)}_p20' if ok else None


if __name__ == '__main__':
    doc = json.load(open(sys.argv[1]))
    s = pick(doc)
    if s is None:
        print('NO_CELL'); sys.exit(3)
    print(s)
