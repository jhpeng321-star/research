"""Synth-only zero-SUMO check: the reference specs the admission root rebuilds == the specs the reference root runs."""
import json, subprocess, sys
sys.path[:0] = ['ops', '.']
import a96_p20adm as M
from pathlib import Path
which, root, ref = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
M._RT['which_ref'] = 'IDENT_U0'
if which == 'ref':
    M.repoint_ref(root.parent)
    specs = M.ref_context(root)[4][0][2]
else:
    M.repoint_adm(root.parent, ref, None)
    pins = M.merged_pins(root)
    M.repoint_adm(root.parent, ref, pins)
    D, panel = M.M76.setup('v', root)
    specs = M.MA.admit_waves(D, panel, root, M.comparator(root), M.MA.candidates(root))[0][2]
print(json.dumps(specs, sort_keys=True, default=str))
