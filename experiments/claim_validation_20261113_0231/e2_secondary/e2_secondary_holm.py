"""E2 registered secondaries + Holm over the two registered primaries (zero SUMO, analysis only).

Wrapper around the UNMODIFIED registered e2_analyse.analyse() (sha256 2cbeb588...). Differences from e2_analyse.main:
  1. output goes to a NEW directory given on the command line (never out/e2/E2_ANALYSIS.json);
  2. the Holm step uses the registered family from PROTOCOL.json multiplicity.primaries:
     'E1 admitted count' p read from out/e1/bank/FROZEN_BANKS.json primary_admitted_count.permutation.p_two_sided,
     'E2 F - H* delay'  p from the recomputed primary (F - H*), i.e. what `--primaries` would have done.
Writes E2_ANALYSIS_REPRO.json (same body as e2_analyse without holm; must reproduce the sealed doc_sha256) and
E2_HOLM.json.
"""
import json
import sys
from pathlib import Path

B = Path('/root/cv_20261113_0231')
sys.path.insert(0, str(B/'cv'))
import cvlib as L  # noqa: E402
import stats as S  # noqa: E402
import e2_analyse as E  # noqa: E402

out_dir = Path(sys.argv[1])
O = B/'cv'/'out'
bank = B/'inputs'/'A42_BIG_BANK_D2800_p60.json'
res_path = O/'e2'/'E2C_RESULTS.json'
sel_path = O/'e2'/'E2_SELECTION.json'

man = L.verify_seal(L.read_json(O/'e2'/'E2_MANIFEST.json'))
specs = {x['family']: x['strategy'] for x in L.bank_strategies(L.read_json(bank))}
for name in man['banks']:
    specs.update({x['family']: x['strategy'] for x in L.bank_strategies(L.read_json(O/'e2'/'banks'/name))})
sel = L.verify_seal(L.read_json(sel_path))
if sel['manifest_sha256'] != man['doc_sha256']:
    raise L.Refused('selection belongs to another manifest')
body = E.analyse(man, L.read_json(res_path), specs, sel['selected_arm'])
body['selection_sha256'] = sel['doc_sha256']
repro = L.seal(dict(schema='claim_validation.e2.analysis.v1', run_id=L.RUN_ID, manifest_sha256=man['doc_sha256'],
                    results_sha256=L.sha_file(res_path), **body))
sealed = L.read_json(O/'e2'/'E2_ANALYSIS.json')

fb_path = O/'e1'/'bank'/'FROZEN_BANKS.json'
fb = L.verify_seal(L.read_json(fb_path))
p_e1 = float(fb['primary_admitted_count']['permutation']['p_two_sided'])
p_e2 = float(body['primary']['p_two_sided'])
ps = {'E1 admitted count': p_e1, 'E2 F - H* delay': p_e2}
adj = S.holm(ps)
holm = L.seal(dict(schema='claim_validation.holm.v1', run_id=L.RUN_ID, method='Holm', alpha=0.05, two_sided=True,
                   family='PROTOCOL.json multiplicity.primaries', raw=ps, adjusted=adj,
                   rejected={k: bool(adj[k] <= 0.05) for k in adj},
                   sources={'E1 admitted count': dict(file=str(fb_path), file_sha256=L.sha_file(fb_path),
                                                      doc_sha256=fb['doc_sha256']),
                            'E2 F - H* delay': dict(file=str(res_path), file_sha256=L.sha_file(res_path),
                                                    analysis_doc_sha256=repro['doc_sha256'])},
                   repro_matches_sealed_E2_ANALYSIS=bool(repro['doc_sha256'] == sealed['doc_sha256'])))
print(L.write_once(out_dir/'E2_ANALYSIS_REPRO.json', L.text_of(repro)))
print(L.write_once(out_dir/'E2_HOLM.json', L.text_of(holm)))
print(json.dumps(dict(repro_doc=repro['doc_sha256'], sealed_doc=sealed['doc_sha256'], raw=ps, adjusted=adj)))
