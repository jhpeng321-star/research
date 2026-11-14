"""Emit explicit permitted pilot bindings from INDEX metadata; never scan runs."""
import argparse
import json
from pathlib import Path

from .guard import DataGuard


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--campaign-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--guard-spec", required=True)
    p.add_argument("--guard-tool", required=True)
    p.add_argument("--allowed-root", action="append", required=True)
    a = p.parse_args(argv)
    guard = DataGuard(a.guard_spec, a.guard_tool, a.allowed_root)
    guard.check_path(a.campaign_root)
    guard.check_path(a.out)
    campaign = Path(a.campaign_root)
    if campaign.name != "campaign_walls_pilot_chunk4":
        raise ValueError("only campaign_walls_pilot_chunk4 is allowed")
    bindings = {}
    for row in guard.read_json(a.index)["runs"]:
        if str(row.get("split")).upper() != "TRAIN" or row.get("policy") != "wall":
            continue
        if Path(str(row.get("root_rel", ""))).name != campaign.name:
            continue
        guard.check_record(row)
        name = row.get("run") or row["key"].rsplit("/", 1)[-1]
        if Path(name).name != name:
            raise ValueError("run must be a basename")
        path = campaign / name
        guard.check_path(path)
        if row["key"] in bindings:
            raise ValueError("duplicate INDEX key")
        bindings[row["key"]] = str(path)
    with Path(a.out).open("x", encoding="utf-8") as fh:
        json.dump(bindings, fh, indent=2)
    print(json.dumps({"n_bindings": len(bindings), "guard": guard.audit()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
