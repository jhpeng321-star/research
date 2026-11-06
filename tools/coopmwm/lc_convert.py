"""LCEXEC universal converter: verified mined row (coopmwm-mined-move-v1) -> lc_program.FreeProgram.

Every verified probe row of every source (NEWMAP2 / PLANGEN / LOWCAV ``nm2_core`` hold rows, PLANGEN2
``pg2_core`` hold + lane rows, COOPKINDS ``ck_core`` lane rows, LCMOVES ``lcm_core`` prog rows) converts to
ONE free_program in the WM frame of the row's map (N1 descriptor, ``lc_program.frame_from_descriptor``):

* members = the row's concrete vids in ``row['members']`` order (the probe's order, initiator first),
  ``join_el`` all 0 (every probe member started at t0), ``duration_s`` = the row's;
* lane-family rows (COOPKINDS / PLANGEN2 lane_balance, early_merge, zipper, zipper_order, discharge, incl.
  R_random ``r_kind`` lane families) go through N9's :func:`lc_convert_probe.probe_free_program` (exact
  role binding + command-log validation);
* hold / pulse / prog / legacy rows (member spec without ``role`` = nm2 ``hold``) go through
  :func:`lc_roles.member_program` on the member spec EXACTLY as the probe Arm received it;
* dither vehicles (non-member ``pulse`` CRN perturbations) are NOT part of the move: excluded and noted
  in ``source['dither_excluded']``;
* anything that cannot be represented exactly raises :class:`NotRepresentable` (``.reason``); a row is
  never projected onto a legacy family (``moves.from_plan`` is not used).

``labels(row, descriptor)`` = per-CAV per-control-second ``lc_program.tick_label`` dicts of the
log-consistent parity run (``lc_parity.run(stream='log')``: the probe Arm reproduces the row's own log
on that stream and our executors + the real env writer issue the same TraCI calls).

CLI::

    python -m tools.coopmwm.lc_convert convert ROWS.jsonl[.gz] OUT.jsonl     # one record per row
    python -m tools.coopmwm.lc_convert labels  ROWS.jsonl[.gz] OUT.jsonl     # one label per line
"""
from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

from . import lc_convert_probe as ncp
from .lc_program import FreeProgram, check_registered, frame_from_descriptor
from .lc_roles import member_program

CONVERTER = 'lcexec-cv-v1'
MINED_SCHEMA = 'coopmwm-mined-move-v1'
REPO = Path(__file__).resolve().parents[2]
MAP_DIRS = {'wz3to2': REPO/'assets/research/newmap_candidates/wz3to2'}

# verification-id prefix -> probe source (row files: NEWMAP_/PLANGEN_/PLANGEN2_/COOPKINDS_/LOWCAV_/LCMOVES_MOVES_*)
PREFIX_SOURCE = {'nm2v': 'newmap2', 'pgv': 'plangen', 'pg2v': 'plangen2', 'ckv': 'coopkinds', 'lcv': 'lowcav',
                 'lcmv': 'lcmoves'}
FILE_SOURCE = {'NEWMAP': 'newmap2', 'PLANGEN': 'plangen', 'PLANGEN2': 'plangen2', 'COOPKINDS': 'coopkinds',
               'LOWCAV': 'lowcav', 'LCMOVES': 'lcmoves'}
# probe Arm roles per source core (verbatim class Arm): nm2_core hold/pulse; pg2_core + lcm_core add the
# COOPKINDS roles (lcm_core also 'prog'); ck_core has its OWN 'hold' (no release_x default, no anchor).
SOURCE_ROLES = {
    'newmap2': ('hold', 'pulse'), 'plangen': ('hold', 'pulse'), 'lowcav': ('hold', 'pulse'),
    'plangen2': ('hold', 'pulse') + ncp.LANE_ROLES,
    'coopkinds': ('pulse',) + ncp.LANE_ROLES,
    'lcmoves': ('hold', 'pulse', 'prog') + ncp.LANE_ROLES,
}


class NotRepresentable(ValueError):
    """The row cannot be represented EXACTLY as a free_program (never projected)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = str(reason)


def move_of(row):
    move = row.get('move')
    return move if isinstance(move, dict) else json.loads(move)


def file_source(file):
    """Probe source of a row file name (``PLANGEN2_MOVES_B.jsonl.gz`` -> 'plangen2'), or None."""
    if not file:
        return None
    stem = Path(str(file)).name.split('.')[0]
    head = stem.split('_MOVES')[0]
    return FILE_SOURCE.get(head)


def source_of(row, file=None):
    """Probe source of a row: from the file name when given (must agree with the id prefix)."""
    by_id = PREFIX_SOURCE.get(str(row.get('verification_id') or '').split('_')[0])
    by_file = file_source(file)
    if by_file is not None and by_id is not None and by_file != by_id:
        raise NotRepresentable(f'row file source {by_file!r} differs from its verification id source {by_id!r}')
    src = by_file or by_id
    if src is None:
        raise NotRepresentable(f'unknown probe source for {row.get("verification_id")!r}')
    return src


def dither_ids(row):
    members = set(row.get('members') or ())
    return [d['id'] for d in (row.get('dither') or []) if d.get('id') not in members]


def member_specs(row):
    """Member specs in ``row['members']`` order (role defaults to the nm2 'hold')."""
    move = move_of(row)
    spec = {m['id']: m for m in move.get('members') or ()}
    members = list(row.get('members') or ())
    if not members or len(set(members)) != len(members):
        raise NotRepresentable('row members must be distinct, non-empty vehicle ids')
    if set(spec) != set(members) or len(spec) != len(move['members']):
        raise NotRepresentable('row members differ from the move member spec')
    lanes = row.get('member_lanes')
    if lanes is not None and [int(x) for x in lanes] != [int(spec[v]['lane']) for v in members]:
        raise NotRepresentable('row member_lanes differ from the move member spec')
    return [dict(spec[v], role=spec[v].get('role') or 'hold') for v in members]


def row_key(row):
    """Unique row identity: verification id + replicate (rows of one verification share the id)."""
    return f"{row.get('verification_id')}#r{row.get('seed')}"


def row_meta(row, file=None):
    """Identity / provenance of one mined row (no outcomes)."""
    move = move_of(row)
    world = row.get('world') or {}
    src = source_of(row, file)
    return dict(key=row_key(row), file=None if file is None else Path(str(file)).name, probe=src,
                arm=row.get('arm'), verification_id=row.get('verification_id'), replicate=row.get('seed'),
                plan_sha256=row.get('plan_sha256'), family=row.get('family'), variant=row.get('variant'),
                r_kind=row.get('r_kind'), t0=row.get('t0'), world_seed=world.get('seed'), cell=world.get('cell'),
                map=move.get('map'), members=list(row.get('members') or ()),
                roles=[m.get('role') or 'hold' for m in member_specs(row)],
                duration_s=row.get('duration_s'), dither_excluded=dither_ids(row))


def _source(row, file, extra=None):
    meta = row_meta(row, file)
    out = dict(schema=MINED_SCHEMA, converter=CONVERTER, file=meta['file'], probe=meta['probe'], arm=meta['arm'],
               verification_id=meta['verification_id'], replicate=meta['replicate'],
               plan_sha256=meta['plan_sha256'], family=meta['family'], variant=meta['variant'],
               r_kind=meta['r_kind'], t0=meta['t0'], world_seed=meta['world_seed'], cell=meta['cell'],
               dither_excluded=meta['dither_excluded'])
    out.update(extra or {})
    return out


def from_mined_row(row, descriptor, *, file=None):
    """EXACT verified mined row -> lc_program.FreeProgram, else :class:`NotRepresentable`."""
    if row.get('schema') != MINED_SCHEMA:
        raise NotRepresentable('not a coopmwm-mined-move-v1 row')
    try:
        frame = frame_from_descriptor(descriptor)
    except ValueError as exc:
        raise NotRepresentable(str(exc)) from exc
    move = move_of(row)
    if move.get('map') != frame.name:
        raise NotRepresentable(f'row map {move.get("map")!r} is not the descriptor map {frame.name!r}')
    src = source_of(row, file)
    duration = row.get('duration_s')
    if type(duration) is not int or duration <= 0:
        raise NotRepresentable('row duration_s must be a positive integer')
    specs = member_specs(row)
    allowed = SOURCE_ROLES[src]
    for m in specs:
        if m['role'] not in allowed:
            why = ('ck_core hold role semantics differ from the nm2 hold (no release_x default, no anchor)'
                   if src == 'coopkinds' and m['role'] == 'hold' else f'role {m["role"]!r} not in the {src} probe Arm')
            raise NotRepresentable(f'{m["id"]}: {why}')
    if ncp.family_of(row) is not None:
        if src not in ('coopkinds', 'plangen2'):
            raise NotRepresentable(f'lane-family row from source {src!r} (N9 binds COOPKINDS/PLANGEN2 only)')
        try:
            program = ncp.probe_free_program(row, frame, file=None if file is None else Path(str(file)).name)
        except ValueError as exc:
            raise NotRepresentable(f'lane-family row: {exc}') from exc
        extra = dict(path='lc_convert_probe', probe_family=program.source.get('probe_family'))
        return dataclasses.replace(program, source=_source(row, file, extra))
    if any(m['role'] in ncp.LANE_ROLES for m in specs):
        raise NotRepresentable('lane roles in a row whose family is not a registered lane family')
    try:
        programs = tuple(member_program(m['role'], m['params'], frame) for m in specs)
    except (KeyError, TypeError, ValueError) as exc:
        raise NotRepresentable(f'member spec: {exc!r}') from exc
    return FreeProgram(tuple(m['id'] for m in specs), programs, map=frame.name, join_el=(0,)*len(specs),
                       duration_s=int(duration), source=_source(row, file, dict(path='lc_roles')))


def registered_violations(program):
    """check_registered as JSON-able records (generator contract; converted rows should be clean)."""
    return [dict(member=program.members[i], key=k, value=v) for i, k, v in check_registered(program)]


def descriptor_for(row):
    from . import map_descriptor as md
    name = move_of(row).get('map')
    if name not in MAP_DIRS:
        raise NotRepresentable(f'no registered lane-drop descriptor for map {name!r}')
    return md.load(MAP_DIRS[name])


def labels(row, descriptor, *, file=None, program=None):
    """Per-CAV per-control-second tick labels (lcexec-tick-label-v1) of the log-consistent parity run."""
    from . import lc_parity
    program = program if program is not None else from_mined_row(row, descriptor, file=file)
    frame = frame_from_descriptor(descriptor)
    res = lc_parity.run(row, program, frame=frame, stream='log', desc=descriptor, file=file, collect_labels=True)
    return res['labels']


# ---------------------------------------------------------------- CLI
def read_rows(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


def _write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True)+'\n')


def convert_file(path):
    out = []
    for row in read_rows(path):
        rec = dict(meta=row_meta(row, path))
        try:
            program = from_mined_row(row, descriptor_for(row), file=path)
            rec.update(status='converted', program=program.to_dict(),
                       registered_violations=registered_violations(program))
        except NotRepresentable as exc:
            rec.update(status='refused', reason=exc.reason)
        out.append(rec)
    return out


def label_file(path):
    out = []
    for row in read_rows(path):
        try:
            desc = descriptor_for(row)
            out += labels(row, desc, file=path)
        except NotRepresentable:
            continue
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.coopmwm.lc_convert')
    ap.add_argument('cmd', choices=('convert', 'labels'))
    ap.add_argument('rows')
    ap.add_argument('out')
    a = ap.parse_args(argv)
    records = convert_file(a.rows) if a.cmd == 'convert' else label_file(a.rows)
    _write(a.out, records)
    if a.cmd == 'convert':
        n = sum(r['status'] == 'converted' for r in records)
        print(json.dumps(dict(rows=len(records), converted=n, refused=len(records)-n)))
    else:
        print(json.dumps(dict(labels=len(records))))
    return 0


__all__ = ['NotRepresentable', 'from_mined_row', 'row_meta', 'row_key', 'labels', 'source_of', 'member_specs',
           'dither_ids', 'registered_violations', 'descriptor_for', 'read_rows', 'CONVERTER']

if __name__ == '__main__':
    raise SystemExit(main())
