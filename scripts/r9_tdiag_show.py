import json, sys
for fn in sys.argv[1:]:
    for l in open(fn):
        r = json.loads(l)
        print({k: v for k, v in r.items() if k not in ("zero_mass", "metrics", "batch_keys", "requires_grad", "grad_norms")})
        if r.get("batch_keys"): print("batch keys", r["batch_keys"])
        print(" grad_norms", r["grad_norms"], "req", set(r["requires_grad"].values()))
        print(" metrics", {k: v for k, v in r["metrics"].items() if "zero" in k or "events" in k or "veto" in k or "physical" in k or k.endswith("tau") or k.endswith("_w")})
        for z in r["zero_mass"]: print("  ZM", z)
