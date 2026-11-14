import json
import statistics as st

BASE = r"E:\rd-wt\ws-eval\_out\wms_eval2\wms_run1\EVAL.json"
MC = r"E:\rd-wt\ws-eval\_out\wms_eval2\wms_run1_mergecal\EVAL.json"


def load(p):
    return json.load(open(p))["results"]


def agg(res, arm, sec, metric):
    v = []
    for r in res:
        for _k, b in r["traffic"][arm]["native"].items():
            x = b["sections"][sec][metric]
            if x is not None:
                v.append(x)
    return round(st.mean(v), 2) if v else None


def perbin(res, arm, sec, met):
    out = {}
    for r in res:
        for k, b in r["traffic"][arm]["native"].items():
            x = b["sections"][sec][met]
            if x is not None:
                out.setdefault(k, []).append(x)
    return {k: round(st.mean(v), 2) for k, v in sorted(out.items(), key=lambda x: int(x[0]))}


base, mc = load(BASE), load(MC)
for sec, met in (("merge", "merge_queue_length"), ("merge", "stopped_vehicles"),
                 ("merge", "mean_speed"), ("merge", "total_delay"),
                 ("upstream", "total_delay"), ("downstream", "total_delay")):
    print("%-11s %-20s base=%8s mergecal=%8s rec=%8s" % (
        sec, met, agg(base, "predicted", sec, met),
        agg(mc, "predicted", sec, met), agg(mc, "recorded", sec, met)))
for name, res in (("base", base), ("mc", mc), ("rec", mc)):
    print("merge_queue", name, perbin(res, "recorded" if name == "rec" else "predicted",
                                     "merge", "merge_queue_length"))
