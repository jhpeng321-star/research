# Fresh-session adoption — MAIN23:32 label-launcher fault

- Read all 554 lines of `coopmwm_prompt.md`, git status/log, current handoff,
  design, and live receipts. Adopt HEAD `da7d24ef` and all inherited changes.
- cpuQ collector supervisor PID1485245/start64300804 is live; its environment
  has neither SUMO_HOME nor the installed SUMO bin directory on PATH.
- The installed binary exists at
  `/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo/bin/sumo`.
- Two completed searches (986016 and 986018, t300) have failed labels: all nine
  branches report SUMO binary not found, actual_sim_s=0. These are void/no-outcome
  attempts, not evidence about plan quality. Preserve their charges and files.
- Adopt gpuA7429/7430 (eight exact gates each, one search finished each),
  gpuR494955 (eight gates, active search), gpuR500795 (three gates, qualifying).
  No GPU restart, search replay, or training dispatch in this adoption.
- A48/B3/R3; 72 episodes and 14,534 snapshots already complete. No u4 update.
- At unix17936338xx the main ledger charged 3,906,810 sim-s. Even adding the
  retained 811,640 superseded engineering allocation gives 4,718,450, below
  the fresh-message 6,000,000 limit. Recovery must remain within this tighter
  limit while the brief/canonical 14,500,000 authorization is clarified.
- Fix the read-only status parser's duplicate `key` keyword, which prevented
  observation as soon as a real broker completion appeared.
- Next: CPU-qualified explicit SUMO environment, policy-branch startup smoke,
  immutable new label attempts, registered charges, grading-only handover.
