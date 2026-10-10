import json

d = json.load(open("wm_api/examples/example_response.json", encoding="utf-8"))
print("api_version", d["api_version"], "| horizon_s", d["horizon_s"],
      "| tau_s", d["tau_s"])
print("members", d["members"], "| arrivals_enabled", d["arrivals_enabled"])
print("checkpoint sha_verified", d["checkpoint"]["sha_verified"],
      "| files", sorted(d["checkpoint"]["sha256"]))
for p in d["plans"]:
    m = p["metrics"]["delay_s_per_veh"]
    q05, q95 = m["quantiles"]["q05"], m["quantiles"]["q95"]
    draws = [round(x, 1) for x in m["member_draws"]]
    pdm = p["paired_diff_vs_plan0"]["delay_s_per_veh"]
    print("plan %d: ref=%s desc=%r" % (p["plan_index"], p["is_reference"],
                                       p["descriptor"][:44]))
    print("   delay q05..q95: %.1f .. %.1f  mean %.2f  draws %s"
          % (q05, q95, m["mean"], draws))
    print("   ood %.4f  dDelay_vs_p0 mean %+.2f  q50 %+.2f"
          % (p["ood_score"], pdm["mean"], pdm["quantiles"]["q50"]))
    if p["wall_gates"]:
        refusal = p["wall_gates"]["pair_gate"]["refusal"]
        print("   wall_gates pair refusal:", refusal)
print("meta.timing_s", d["meta"]["timing_s"], "| k_slots",
      d["meta"]["k_slots"], "| validity", d["meta"]["validity"])
