"""Hard filesystem boundary, inherited by every Python scoring/training child."""
import os
from pathlib import Path
import sys

BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vtype')


def safe(path):
    text = os.fsdecode(path)
    if any(t in text.lower() for t in BAD):
        raise PermissionError('S-RAMPFIX restricted path refused')
    resolved = str(Path(text).resolve())
    if any(t in resolved.lower() for t in BAD):
        raise PermissionError('S-RAMPFIX restricted target refused')
    return Path(text)


def install():
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            safe(args[0])
        if event == 'subprocess.Popen':
            executable = Path(os.fsdecode(args[0])).name.lower()
            if executable.startswith('sumo'):
                raise PermissionError('ZERO SUMO')
    sys.addaudithook(audit)
