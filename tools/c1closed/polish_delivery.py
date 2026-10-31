"""Presentation-only repair from verified CSVs; no forecasts or re-analysis."""
from pathlib import Path
import csv
import hashlib
import json
from unittest.mock import patch

import analyze
import verify_delivery

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/'docs/paper_results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt_path = OUT/'C1CLOSED_ANALYSIS.json'
    old = json.loads(receipt_path.read_text())
    assert old['analyzer_sha256'] == digest(HERE/'analyze.py')
    assert 'presentation_revision' not in old
    for name, expected in old['outputs'].items():
        assert digest(OUT/name) == expected
    assert json.loads((HERE/'evidence/DELIVERY_VERIFIED.json').read_text())['passed']
    assert (HERE/'out').is_dir()
    archive = HERE/'out/presentation_before'
    archive.mkdir()
    for source in (receipt_path, OUT/'C1CLOSED.md', OUT/'C1CLOSED_error_horizon.png',
                   HERE/'evidence/DELIVERY_VERIFIED.json'):
        with (archive/source.name).open('xb') as f:
            f.write(source.read_bytes())
    metric_names = ('C1CLOSED_per_world.csv','C1CLOSED_paired.csv','C1CLOSED_bootstrap.csv')
    metric_hashes = {n:digest(OUT/n) for n in metric_names}
    with (OUT/'C1CLOSED_bootstrap.csv').open(encoding='utf-8',newline='') as f:
        summary = list(csv.DictReader(f))
    for row in summary:
        row['horizon_s'] = int(row['horizon_s'])

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    savefig = Figure.savefig

    def spaced_footer(fig, *args, **kwargs):
        footers = [t for t in fig.texts if t.get_text().startswith('Endpoint gaps =')]
        assert len(footers) == 1
        footers[0].set_position((.5,-.065))
        footers[0].set_verticalalignment('top')
        return savefig(fig,*args,**kwargs)

    target = HERE/'out/presentation_after'
    target.mkdir()
    # Reuse the exact frozen plot and CSV values; only footer placement changes.
    with patch.object(Figure,'savefig',spaced_footer):
        analyze.plot(summary,target)
    (OUT/'C1CLOSED_error_horizon.png').write_bytes((target/'C1CLOSED_error_horizon.png').read_bytes())
    report = (OUT/'C1CLOSED.md').read_text(encoding='utf-8')
    report += ('## Interpretation and numerical limitation\n'
        'At 60 s neither PI nor MAPPO paired HDV-position contrast excludes zero; current closed-loop endpoint superiority is inconclusive.\n'
        'Lower 120–300 s team-delay error versus BB-traj coexists with large absolute errors and weak selection; it does not establish accurate controller evaluation.\n'
        'Mixed-hardware exact cross-host parity failed; the prospectively registered numerical/ensemble gate passed 158/158. See `C1CLOSED_RUNTIME.md`; this is not a SUMO divergence floor.\n')
    assert len(report.splitlines()) <= 60
    (OUT/'C1CLOSED.md').write_text(report,encoding='utf-8')
    assert {n:digest(OUT/n) for n in metric_names} == metric_hashes
    updated = dict(old)
    updated['outputs'] = {n:digest(OUT/n) for n in old['outputs']}
    updated['presentation_revision'] = dict(script_sha256=digest(Path(__file__)),
        original_analysis_sha256=digest(archive/receipt_path.name),
        original_plot_sha256=old['outputs']['C1CLOSED_error_horizon.png'],
        original_scientific_commit='b40939aa',metric_files_unchanged=metric_hashes,
        changes='footer position and explicit scope interpretation only; frozen analyzer/inference/CSV values unchanged')
    receipt_path.write_text(json.dumps(updated,indent=1),encoding='utf-8')
    # Retain the original successful verification and generate a new one for the
    # actual changed report/PNG/manifest using the unchanged verifier.
    (HERE/'evidence/DELIVERY_VERIFIED.json').rename(HERE/'evidence/DELIVERY_PRE_LAYOUT.json')
    verify_delivery.main()
    check = dict(passed=True,metric_files_unchanged=metric_hashes,
        analyzer_unchanged=digest(HERE/'analyze.py') == old['analyzer_sha256'],
        original_archive=str(archive),analysis_sha256=digest(receipt_path),
        script_sha256=digest(Path(__file__)),
        changes=updated['presentation_revision']['changes'])
    with (HERE/'evidence/PRESENTATION_UPDATED.json').open('x') as f:
        json.dump(check,f,indent=1)
    print(json.dumps(check))


if __name__ == '__main__':
    main()
