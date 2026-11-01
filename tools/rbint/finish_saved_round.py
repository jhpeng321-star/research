"""Explicit CPU-analysis recovery; reuse every successful saved component.

Never fetch new forecasts, dispatch inference, alter a failed output, or retry
automatically. The caller diagnoses an exited collector and supplies its STOP.
"""
import argparse
from datetime import datetime
from pathlib import Path

from tools.rbint.continue_noon import (ROOT, PUBLISH, VARIANTS, command, fingerprint,
    state_from_publication, update_handoff)
from tools.rbint.noon_report import render
from tools.rbeval.core import digest, load, save_new


def reusable_component(directory, name, artifact, previous, same_ensemble):
    receipt = directory/(name+'_EXIT.json')
    if receipt.is_file() and load(receipt)['returncode'] == 0:
        assert artifact.exists(), 'successful component artifact missing'
        return str(artifact)
    return previous if same_ensemble and previous else None


def saved_snapshot(directory, key, previous):
    process = directory/('snapshot_'+key+'_PROCESS.json')
    if not process.exists():
        # A failure may precede later variants' transfer. Reuse their ACTUAL
        # older snapshot/receipt rather than relabeling it as a new collection.
        return Path(previous['snapshot']), True
    args = load(process)['argv']
    assert load(directory/('snapshot_'+key+'_EXIT.json'))['returncode'] == 0
    return Path(args[args.index('--out')+1]), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--stop', required=True)
    args = parser.parse_args()
    directory = Path(args.round).resolve()
    assert directory.parent == PUBLISH and not (directory/'PUBLICATION.json').exists()
    stop = load(args.stop)
    assert stop['automatic_retry'] is False and stop['remote_inference_dispatched'] is False
    failures = [prefix+'_'+key+' failed:' for prefix in ('compare','distribution','credibility','endpoints') for key in VARIANTS]
    assert any(name in stop['error'] for name in failures), 'explicit failed analysis component required'
    save_new(directory/'EXPLICIT_CPU_RECOVERY.json', dict(stop=args.stop,
        reason='explicit failed CPU-analysis component exit; cause unresolved; successful saved components are never repeated',
        automatic_retry=False, forecast_bytes_unchanged=True, inference_dispatched=False,
        recovery_time=datetime.now().astimezone().isoformat()))
    previous = load(args.previous)
    state = state_from_publication(args.previous)
    publication = dict(previous, local_time=datetime.strptime(directory.name,'%Y%m%dT%H%M%S%z').isoformat(),
        source_head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        variants={}, receipt_directory=str(directory.relative_to(ROOT)).replace('\\','/'),
        explicit_cpu_recovery=True, code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (
            Path(__file__), ROOT/'tools/rbint/compare.py', ROOT/'tools/rbint/noon_report.py')})
    baseline, endpoint_baseline = state['baseline'], state['endpoint_baseline']
    for key in VARIANTS:
        old = previous['variants'].get(key, {})
        snapshot, older = saved_snapshot(directory,key,old)
        info = fingerprint(snapshot)
        item = dict(info, snapshot=str(snapshot))
        same = old.get('sha256') == info['sha256']
        if older:
            assert same, 'prior snapshot bytes changed'
            item['reused_older_snapshot_not_collected_before_stop'] = True
        report = directory/key
        report.mkdir(exist_ok=True)
        components = [
            ('comparison', 'compare', 'paired/COMPARISON.json', ['-m','tools.rbint.compare','--progress','--candidate',snapshot,'--baseline',baseline]),
            ('distribution', 'distribution', 'distribution', ['-m','tools.rbeval.score','--inputs',snapshot,'--allow-partial']),
            ('credibility', 'credibility', 'credibility', ['-m','tools.wmdecide','credibility','--eval-root',ROOT,'--inputs',snapshot,'--allow-partial'])]
        if key.startswith('FULL-'):
            components.append(('endpoints','endpoints','endpoints/HDV60_COMPARISON.json',
                ['-m','tools.rbint.score_endpoints','--candidate',snapshot,'--baseline',endpoint_baseline,
                 '--factual',ROOT/'docs/rbint/hdv60/FACTUAL_065e9853.json']))
        for field, prefix, suffix, argv in components:
            artifact = report/suffix
            reuse = reusable_component(directory, prefix+'_'+key, artifact, old.get(field), same)
            if reuse:
                item[field] = reuse
                continue
            assert info['complete'], 'do not manufacture incomplete-K scores'
            # A distinct destination/log retains the failed component exactly.
            out = report/(Path(suffix).parts[0]+'_recovered')
            assert not out.exists(), 'explicit recovery is one-use, never automatic retry'
            command([*argv,'--out',out],directory,prefix+'_'+key+'_recovered')
            item[field] = str(out/Path(suffix).name) if Path(suffix).suffix else str(out)
        publication['variants'][key] = item
        save_new(directory/('RECOVERED_'+key+'.json'), item)
    save_new(directory/'PUBLICATION.json',publication)
    (ROOT/'docs/paper_results/WMPHYS_MORNING.md').write_text(render(publication),encoding='utf-8')
    handoff = ROOT/'HANDOFF.md'
    block = ('## Bounded noon collector\nExplicit saved-component CPU recovery completed; no new inference.\n'
        f"Latest publication `{publication['receipt_directory']}/PUBLICATION.json`;08interim retained.\n"
        'All successful components reused; only previously unfinished analysis executed.\n'
        'Next bounded collector resumes this committed publication/cache;09:30freeze/11:45report unchanged.')
    handoff.write_text(update_handoff(handoff.read_text(encoding='utf-8'),block),encoding='utf-8')
    print(publication['receipt_directory'],flush=True)


if __name__ == '__main__':
    main()
