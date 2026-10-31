# S-RB-EVAL handoff

- Branch c4/rb-eval, base f0db3c82; fresh-session audit found clean tree and no
  previous RBEVAL implementation/dispatch receipts. Existing WMCALIB cost-only
  tasks lack trajectories and native predictions; they cannot fill this panel.
- Frozen design: docs/paper_results/RBEVAL_DESIGN.md. 20 worlds 969420–969439,
  all five arms, t0=300, K8, horizons 120/300/600. Small-cell precision prohibits
  qualification; retain separate diagnostic and gate verdicts.
- Implementation in progress under tools/rbeval; shared model files untouched.
- gpuR first resource check: GPU 1 MiB/24564 MiB, utilization 0%; host load 18.84.
  No matching RBEVAL process seen. No inference launched yet.
- Reuse S-WMBREAKDOWN observable receipts and registered AST-extracted LRDIAG
  detector. Helpers rb-shared/targets and obsstate will be consumed if available.
