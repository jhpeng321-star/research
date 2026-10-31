"""Run the prepared companion tooling against a hash-verified partial EXAM view."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from .protocol import checked, sha
from .storage import read, write


def analyze(exam, output, helper_root):
    exam, output, helper_root = map(checked, (exam, output, helper_root))
    manifest = read(exam/'MANIFEST.json')
    for name, expected in manifest.items():
        if Path(name).name != name or sha(checked(exam/name)) != expected:
            raise ValueError('EXAM manifest mismatch: '+name)
    # The companion loader requires selection-world equality even for partial
    # data. Preserve the original export; derive a documented metadata-only view.
    view = output.parent/'helper-input'
    view.mkdir()
    for name in manifest:
        if name not in ('worlds.json', 'provenance.json'):
            shutil.copyfile(checked(exam/name), checked(view/name))
    states = [json.loads(line) for line in (exam/'states.jsonl').read_text().splitlines()]
    seeds = {s['world_id'] for s in states}
    selection = read(exam/'worlds.json')
    selection['worlds'] = [w for w in selection['worlds'] if w['world']['seed'] in seeds]
    selection['solo_worlds'] = [s for s in selection['solo_worlds'] if s in seeds]
    write(view/'worlds.json', selection)
    provenance = read(exam/'provenance.json')
    provenance['companion_view'] = dict(parent_manifest_sha256=sha(exam/'MANIFEST.json'),
        operation='worlds.json membership restricted to represented partial worlds; labels/features unchanged')
    write(view/'provenance.json', provenance)
    write(view/'MANIFEST.json', {name: sha(view/name) for name in manifest})
    sys.path.insert(0, str(helper_root))
    from tools.coopbandit_learn.cli import analyze as helper
    return helper(view, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exam')
    parser.add_argument('output')
    parser.add_argument('helper_root')
    args = parser.parse_args()
    # tools is an existing regular package: extend its search path explicitly.
    import tools
    tools.__path__.append(str(Path(args.helper_root)/'tools'))
    analyze(args.exam, args.output, args.helper_root)
