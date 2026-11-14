"""Observable committed-plan context, shared by plant and WM decoders."""
from .plans import acceleration_at


def members(commitments):
    return {v for c in commitments for v in c['plan']['members']}


def background(commitments, elapsed_s):
    out={}
    for c in commitments:
        values=acceleration_at(c['plan'],c['elapsed_s']+elapsed_s)
        if set(out)&set(values):raise ValueError('overlapping committed plan members')
        out.update(values)
    return out


def validate(commitments, group=()):
    seen=set(group)
    for c in commitments:
        if set(c)!={'plan','elapsed_s'}:raise ValueError('committed context whitelist')
        p=c['plan'];vs=set(p['members'])
        if seen&vs:raise ValueError('new group overlaps committed members')
        if not 0<=c['elapsed_s']<p['duration_s']:raise ValueError('expired committed context')
        seen.update(vs)
    return commitments
