"""Clean4 pre-launch POTENTIAL gate (MAIN 2026-11-06, registered before launch).

Training may start only if the N3/N8 reports show, in the chosen training cell:
  (1) robust admissible group synergy (per-seed CI95 low > 0, later-harm and dither
      checks pass) at >= 15 % of screened bottleneck moments;
  (2) at least 8 verified B examples;
  (3) at least 50 % of those examples reachable within the configured V2V/RSU range.
Otherwise: do NOT launch; write CLEAN4_GATE_FAIL.md with the numbers and stop.

Pure and zero-physics: the operator extracts the counts from the producers' reports
(cited by path + SHA) and this module computes the verdict and seals the receipt.
"""
from .exact_grade import _seal, verify_sealed

RULE = ('MAIN 2026-11-06 pre-launch potential gate: in the chosen training cell (1) robust admissible '
        'group synergy (per-seed CI95 low>0, later-harm and dither checks pass) at >=15% of screened '
        'bottleneck moments; (2) >=8 verified B examples; (3) >=50% of those examples reachable within '
        'the configured V2V/RSU range. Fail -> no launch, CLEAN4_GATE_FAIL.md, stop.')
MIN_ROBUST_SHARE = 0.15
MIN_B_EXAMPLES = 8
MIN_REACHABLE_SHARE = 0.50
SCHEMA = 'coopmwm.clean4.potential_gate.v1'


def _int(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer required: '+label)
    return value


def verdict(*, cell, screened_moments, robust_moments, b_examples, sources, range_m):
    """b_examples: list of dicts {id, reachable: bool, robust: bool} (verified B examples in the cell)."""
    screened = _int(screened_moments, 'screened_moments')
    robust = _int(robust_moments, 'robust_moments')
    if robust > screened:
        raise ValueError('robust moments cannot exceed screened moments')
    if not isinstance(cell, dict) or not cell:
        raise ValueError('chosen training cell required')
    if not sources or any(not isinstance(s, dict) or not s.get('path') or len(s.get('sha256', '')) != 64
                          for s in sources):
        raise ValueError('producer reports must be cited by path and SHA-256')
    ids = [e['id'] for e in b_examples]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate B example id')
    if any(not isinstance(e.get('reachable'), bool) for e in b_examples):
        raise ValueError('every B example needs an explicit reachable flag')
    count = len(b_examples)
    reachable = sum(e['reachable'] for e in b_examples)
    share = robust/screened if screened else 0.0
    reach_share = reachable/count if count else 0.0
    checks = dict(robust_share=dict(value=share, min=MIN_ROBUST_SHARE, robust=robust, screened=screened,
                                    passed=screened > 0 and share >= MIN_ROBUST_SHARE),
                  b_examples=dict(value=count, min=MIN_B_EXAMPLES, passed=count >= MIN_B_EXAMPLES),
                  reachable_share=dict(value=reach_share, min=MIN_REACHABLE_SHARE, reachable=reachable,
                                       range_m=range_m, passed=count > 0 and reach_share >= MIN_REACHABLE_SHARE))
    passed = all(c['passed'] for c in checks.values())
    return _seal(dict(schema=SCHEMA, rule=RULE, cell=cell, checks=checks, b_examples=list(b_examples),
                      sources=list(sources), decision='PASS' if passed else 'FAIL', new_sim_s=0),
                 'receipt_sha256')


def require_pass(receipt):
    verify_sealed(receipt, 'receipt_sha256')
    if receipt.get('schema') != SCHEMA or receipt.get('rule') != RULE:
        raise ValueError('registered clean4 potential-gate receipt required')
    if receipt['decision'] != 'PASS' or not all(c['passed'] for c in receipt['checks'].values()):
        raise ValueError('clean4 potential gate did not pass: no launch')
    return receipt


def fail_markdown(receipt):
    c = receipt['checks']
    lines = ['# CLEAN4_GATE_FAIL - pre-launch potential gate failed (no launch)', '',
             'Rule: '+receipt['rule'], '', 'Chosen training cell: '+repr(receipt['cell']), '',
             '| check | value | required | pass |', '|---|---|---|---|',
             '| robust admissible group-synergy share of screened moments | %.3f (%d/%d) | >= %.2f | %s |' % (
                 c['robust_share']['value'], c['robust_share']['robust'], c['robust_share']['screened'],
                 MIN_ROBUST_SHARE, c['robust_share']['passed']),
             '| verified B examples | %d | >= %d | %s |' % (c['b_examples']['value'], MIN_B_EXAMPLES,
                                                          c['b_examples']['passed']),
             '| reachable within V2V/RSU range (%s m) | %.3f (%d/%d) | >= %.2f | %s |' % (
                 c['reachable_share']['range_m'], c['reachable_share']['value'], c['reachable_share']['reachable'],
                 c['b_examples']['value'], MIN_REACHABLE_SHARE, c['reachable_share']['passed']),
             '', 'Sources: '+'; '.join(f"{s['path']} (sha256 {s['sha256']})" for s in receipt['sources']),
             '', 'Receipt sha256: '+receipt['receipt_sha256']]
    return '\n'.join(lines)+'\n'
