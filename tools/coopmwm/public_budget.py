"""Reconcile registered authorities for reporting; never writes the live ledger."""


def reconcile(shared, runtime, runtime_authority):
    """Keep current shared phase caps plus a runtime-only practice transfer.

    The immutable runtime can predate newer bank transfers. Its whole phase
    table must therefore not replace the shared binding's newer phase table.
    The registered practice transfer can also already be in the shared table;
    compare its endpoints before applying it so it is never counted twice.
    """
    if shared['total'] != runtime['total']:
        raise ValueError('public budget authorities have different total caps')
    phases = dict(shared['phases'])
    if any(v < 0 for v in phases.values()) or sum(phases.values()) > shared['total']:
        raise ValueError('invalid shared phase allocation')
    applied = []
    seen = set()
    for transfer in runtime.get('transfers', []):
        source, target, amount = (transfer[k] for k in ('source', 'target', 'amount'))
        digest = transfer['amendment_sha256']
        if (source, target, amount) != ('paired_branch', 'v2v_train', 345600):
            raise ValueError('unrecognised runtime-only transfer; review authority')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid transfer amendment digest')
        if digest not in runtime_authority or digest in seen:
            raise ValueError('unregistered or duplicate practice transfer')
        seen.add(digest)
        endpoints = (phases[source], phases[target])
        declared = (runtime['phases'][source], runtime['phases'][target])
        adjusted = (endpoints[0] - amount, endpoints[1] + amount)
        if endpoints == declared:
            state = 'already_in_shared_binding'
        elif adjusted == declared and adjusted[0] >= 0:
            phases[source], phases[target] = adjusted
            state = 'runtime_only_applied_for_reporting'
        else:
            raise ValueError('practice transfer endpoints conflict with authorities')
        applied.append(dict(amendment_sha256=digest, source=source, target=target,
                            amount=amount, state=state))
    if sum(phases.values()) != sum(shared['phases'].values()):
        raise ValueError('reporting transfer changed the total allocation')
    return dict(total=shared['total'], phases=phases, transfers=applied)
