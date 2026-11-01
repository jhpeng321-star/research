"""Final artifact assembly ONLY: no transfer, inference, fitting or scoring."""
from datetime import datetime
from pathlib import Path
from tools.rbint.continue_noon import ROOT, fingerprint, update_handoff
from tools.rbint.noon_report import render
from tools.rbeval.core import digest, load, save_new


def main():
    directory = ROOT/'docs/rbint/noon/20261101T113555+0800'
    import sys
    if '--render-existing' in sys.argv:
        publication=load(directory/'PUBLICATION.json')
        report=ROOT/'docs/paper_results/WMPHYS_MORNING.md'
        text=render(publication);assert len(text.splitlines())<=60
        report.write_text(text,encoding='utf-8')
        save_new(ROOT/'docs/rbint/FINAL_PRESENTATION.json',dict(
            generated_at=datetime.now().astimezone().isoformat(),
            publication_sha256=digest(directory/'PUBLICATION.json'),
            renderer_sha256=digest(ROOT/'tools/rbint/noon_report.py'),report_sha256=digest(report),
            lines=len(text.splitlines()),new_scoring=False,new_inference=False,
            supersedes_presentation_only='FINAL_ASSEMBLY.json'))
        return
    previous = load(ROOT/'docs/rbint/noon/20261101T105204+0800/PUBLICATION.json')
    publication = dict(previous, label='final',local_time='2026-11-01T11:35:55+08:00',
        receipt_directory=str(directory.relative_to(ROOT)).replace('\\','/'),variants={},
        final_assembly_only=True,assembled_at=datetime.now().astimezone().isoformat())
    for key, old in previous['variants'].items():
        exit_path=directory/f'snapshot_{key}_EXIT.json'
        if not exit_path.exists():
            assert key == 'FULL-v4' and len(old['complete']) == 20
            item=dict(old,reused_verified_complete_frozen_source_after_redundant_transfer_timeout=True)
        else:
            assert load(exit_path)['returncode']==0
            argv=load(directory/f'snapshot_{key}_PROCESS.json')['argv']
            snapshot=Path(argv[argv.index('--out')+1]);info=fingerprint(snapshot)
            item=dict(info,snapshot=str(snapshot))
            for field,prefix,suffix in (
                ('comparison','compare','paired/COMPARISON.json'),
                ('distribution','distribution','distribution'),
                ('credibility','credibility','credibility'),
                *([('endpoints','endpoints','endpoints/HDV60_COMPARISON.json')] if key.startswith('FULL-') else [])):
                done=directory/f'{prefix}_{key}_EXIT.json'
                artifact=directory/key/suffix
                if done.exists() and load(done)['returncode']==0:
                    assert artifact.exists();item[field]=str(artifact)
                else:
                    assert info['sha256']==old['sha256'] and old.get(field)
                    item[field]=old[field]
        publication['variants'][key]=item
    save_new(directory/'PUBLICATION.json',publication)
    report=ROOT/'docs/paper_results/WMPHYS_MORNING.md'
    text=render(publication);assert len(text.splitlines())<=60
    report.write_text(text,encoding='utf-8')
    save_new(ROOT/'docs/rbint/FINAL_ASSEMBLY.json',dict(
        assembled_at=publication['assembled_at'],publication_sha256=digest(directory/'PUBLICATION.json'),
        renderer_sha256=digest(ROOT/'tools/rbint/noon_report.py'),report_sha256=digest(report),
        lines=len(text.splitlines()),new_scoring=False,new_inference=False,
        delayed_final_assembly=True,reason='redundant FULL-v4 transfer exhausted publication deadline; verified complete cache reused'))
    handoff=ROOT/'HANDOFF.md'
    block=('## Final publication\nFinal report assembled from completed score components and verified frozen-source cache.\n'
        f"Publication `{publication['receipt_directory']}/PUBLICATION.json`; FINAL_ASSEMBLY.json records actual delivery time.\n"
        'No remaining collector daemon; no new inference/scoring. Point switching rule FAIL; full-world CIs NA.\n'
        'FULL-v4 complete20; actual A/B/C coverage in final publication. All original failures/partial transfers retained.')
    handoff.write_text(update_handoff(handoff.read_text(encoding='utf-8'),block),encoding='utf-8')
    print(publication['assembled_at'],len(text.splitlines()),flush=True)


if __name__=='__main__': main()
