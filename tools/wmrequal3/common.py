"""Named runtime, guarded inputs, and bounded transport for this task."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OWN = '/root/autodl-tmp/c4/wmrequal3_20261030'
OLD = '/root/autodl-tmp/c4/wmrequal20261030'
sys.path.insert(0, str(ROOT / 'tools/wmrequal'))
from remote import remote

GUARD = r'''
import os, sys
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  name=os.fsdecode(args[0]).lower()
  if any(x in name for x in bad) or name.endswith('.xml'):raise RuntimeError('prohibited input path')
sys.addaudithook(guard)
'''
