"""Mining-side G5 solo execution (operator 2). Applied ONLY to the mining verify source copy.

Accepted G5 (b903c6c2) group plans are unchanged and still pass group_lane.from_plan /
verify_tape. The generic replay_bank.solo_candidate slices per-member matrices, which
leaves a 1-member plan that from_plan rejects (role binding), so G5 solos could never
run. This marks the solo (g5_solo_member) and lets controller_targets issue only that
member's own role (its lane attempt, if any); the other member follows its policy.
"""
import hashlib
from pathlib import Path
import sys

RB_OLD = "    out['id'] = f\"{plan['id']}__solo_{member}\"\n    return out\n"
RB_NEW = ("    out['id'] = f\"{plan['id']}__solo_{member}\"\n"
          "    if plan.get('g5_move'):\n"
          "        out['g5_solo_member'] = member  # mining op2: execute only this member's G5 role\n"
          "    return out\n")
GL_OLD = ("        if plan.get('g5_move'):\n"
          "            move = from_plan(plan)\n"
          "            for vid in move.members:\n")
GL_NEW = ("        if plan.get('g5_move') and plan.get('g5_solo_member'):\n"
          "            solo_move = moves.Move.from_dict(plan['g5_move'])\n"
          "            solo = plan['g5_solo_member']\n"
          "            if plan.get('members') != [solo] or solo not in solo_move.members:\n"
          "                raise ValueError('G5 solo binding mismatch')\n"
          "            requests.pop(solo, None)\n"
          "            requests.update({v: t for v, t in targets(solo_move, now-entry['start_s']).items()\n"
          "                             if v == solo})\n"
          "            continue\n"
          "        if plan.get('g5_move'):\n"
          "            move = from_plan(plan)\n"
          "            for vid in move.members:\n")


def apply(source):
    out = {}
    for rel, old, new in (('tools/coopmwm/replay_bank.py', RB_OLD, RB_NEW),
                          ('tools/coopmwm/group_lane.py', GL_OLD, GL_NEW)):
        p = Path(source)/rel
        text = p.read_text()
        assert text.count(old) == 1, rel
        before = hashlib.sha256(text.encode()).hexdigest()
        text = text.replace(old, new)
        p.write_text(text)
        out[rel] = dict(before=before, after=hashlib.sha256(text.encode()).hexdigest())
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(apply(sys.argv[1])))
