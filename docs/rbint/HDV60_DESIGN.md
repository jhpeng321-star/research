# HDV60 non-regression evidence — frozen before endpoint inference/scoring

- Same registered20starts969420..969439, five arms, K8, t0=300; report
  development12/reserved8 separately. No new SUMO or neural/black-box training.
- Baseline uses the ORIGINAL EVAL WM2 runtime/config/checkpoint on gpuR. Keep
 600s entry padding and1200-step random plans; execute only120 halfsteps.
  One short canary must reproduce original600s task's first six10s regional
  speed/support bins (all5arms,3regions, counts exact, speed tolerance1e-8)
  before continuation. Sources/inputs remain hash-bound; no600s panel repeat.
- New WM-PHYS full forecasts save their60s endpoints directly. Original observed
  string IDs are exported from the hash-verified t0 history, not packed ordinals.
  Predicted exits/recycled slots are censored, never replaced by newborn positions.
- Factual targets use only registered converted observable episode metadata,
  time_s/veh_kin/veh_mask/is_cav/vid arrays. Match t0 identities by an EXACT unique
  observed position/speed/lane-junction/ownership key, then follow converted VID
  to360s. No inferred-driver saved-ID crosswalk. Reject ambiguous/unmatched starts.
  Future absent/junction-carried positions remain censored and counts reported.
- PRIMARY per-ID point prediction = median of itsK8 predicted positions.
  Use a paired common cohort: t0 HDVs with valid factual360s positions and
  uncensored endpoints in allK8 futures of BOTH compared models. Pool the5arms'
  absolute position errors inside each world; take the world's median error.
  Compare candidate-minus-WM2 MEAN of those world medians, in metres; negative
  is better. Also retain sample-mean absolute-error sensitivity and all attrition.
- CIs: registered EVAL10000 whole-world demand×penetration×split-stratified
  bootstrap, seed20261031; saved ensembles/common cohort conditioned upon.
  No vehicle/future pseudo-replication. Degenerate/missing uncertainty stays NA.
- Report point non-regression(mean difference<=0) separately from support by
  its95%CI(upper<=0). Missing worlds/cohorts/uncertainty do not imply qualification.
  This does not replace the separate breakdown/delay/credibility conditions.
