import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    keys = ["step", "loss", "grad_norm", "grad_clipped", "lr", "merge_pivot_tf_events", "merge_pivot_tf_zero_mass_events", "merge_pivot_tf_positive_mass_events",
            "merge_pivot_on_events", "merge_pivot_on_zero_mass_events", "merge_pivot_tau", "merge_pivot_w", "merge_pivot_on_nll", "merge_pivot_tf_nll", "stage_s", "finite"]
    print({k: r.get(k) for k in keys})
