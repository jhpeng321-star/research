import json, sys
from pathlib import Path
from h60_probe import states

ref = list(states(Path(sys.argv[1])))[0]
new = list(states(Path(sys.argv[2])))[0]
rh, nh = ref["history"], new["history"]
print("ref steps", rh["history_steps"], "new steps", nh["history_steps"])
print("ref n_ids", len(rh["ids"]), "new n_ids", len(nh["ids"]))
labels = [str(x) for x in rh["ids"]]
col = {str(x): j for j, x in enumerate(nh["ids"])}
# pick first 3 ref labels
for label in labels[:3]:
    j = col[label]
    rk = [row[labels.index(label)] for row in rh["kin"]]
    nk = [row[j] for row in nh["kin"]]
    print("LABEL", label, "refcol", labels.index(label), "newcol", j)
    print("  ref kin", rk)
    print("  new kin", nk)
    print("  ref mask", [row[labels.index(label)] for row in rh["mask"]])
    print("  new mask", [row[j] for row in nh["mask"]])
    print("  new target", [row[j] for row in nh["target_mask"]])
