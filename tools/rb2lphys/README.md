# Learned physical parameters (owner-approved round 2)

Switch: `PhysicsConfig.physics_params='learned_v1'` plus explicit
`physics_params_file`. The checkpoint/physics defaults keep it **off**.
`configs/wm_phys_lp.toml` is an exact two-field ablation of `wm_phys.toml`.
INT can add those two fields to its FULL-v5 recipe for `FULL-v5-LP`.

The artifact contains three population probabilities: SUMO departLane first
versus free; mainline departSpeed last versus max; ramp departSpeed last versus
max. These sample hidden *physical departure settings*, not events, accelerations
or noise. One source-ordinal draw persists through blocked retries and across
alternative controllers. Insertion safety, source accounting, IDM/LC forms,
the existing joint driver population and causal conditioning remain in force.

Coordination: FIT/LANE own driver IDM and five LC coordinates. This work does not
independently refit them, gap multipliers or impatience clocks. There is no
evidence justifying a context-parameter network in this departure-only screen.
Learned motion/LC/merge corrections, black-box training and SUMO execution are OFF.

## Development evidence and reproduction

- `docs/rb2lphys/PLAN.md` registers scope/objectives before outcome inference.
- `fit_departures.py` uses only six observable files from explicitly resolved
  native969400–969431 tapes. It excludes junction reappearances/ambiguous birth
  order, freezes joint IDM support, and uses16common support draws. Prefix
  240..300s fit; later native development969420–969431300..600s check. Converted
  views stop600s; the originally requested900s upper bound is not actual support.
- RBF-GP expected-improvement search: eight corners + centre +48queries. Loss
  is equal-world/entrance mean normalized birth-speed CRPS +lane/birth Brier.
  This conditional screen alone does not qualify full traffic outcomes.
- `evaluate.py` filters the registered start producer before artifact reads,
  runs the frozen five-arm600s evaluator, plans the identicalK8 latent streams
  and scores its firstK2 as explicitly labelledK2 development evidence.
- `panel.py` reuses each own canary once and runs at most2workers. `compare.py`
  requires every planned task, checks hashes/conservation/CRN, and reports delay,
  BIC onset/rates,10s speed fields,LC counts/time bins and expected point effects.
  Singleton-cell CIs stay unavailable. Ramp-lane loss/source wait are separate;
  they are not mislabelled as full ramp-origin delay.

Existing environments only (one BLAS/OpenMP thread,nice19,resource admission):

```text
python -m tools.rb2lphys.fit_departures --root /home/ubuntu/wmdata20261031/views --out NEW_EXISTING_OUT
python -m tools.rb2lphys.evaluate --world 969420 --futures 2 --params off --out NEW_TASK
python -m tools.rb2lphys.evaluate --world 969420 --futures 2 --params learned_v1 --out NEW_TASK
python -m tools.rb2lphys.compare --panel PANEL.json --targets E:/agent-work/rb-shared/targets --out NEW_EXISTING_OUT
python -m pytest tests/test_rb2lphys.py tests/test_rbint_recipe.py tests/test_rbint_departure.py tests/test_rbint_merge_fast.py -q -p no:cacheprovider
```

All outputs are immutable/new. Exposures include the entire FIT32 native tapes
used by diagnostics and the development12 five-arm comparison. WMDATA2 is
quarantined and reserved969432–969439 are absent from every selection reader.

Hashing: old Windows `git archive` exports contain CRLF. The final artifact and
result attributes pin LF. `verify_delivery.py` records LF/blob hashes, raw
worktree hashes and the ACTUAL historical evaluated archive hashes separately,
and verifies equality of parsed parameters/rule source modulo line endings.
