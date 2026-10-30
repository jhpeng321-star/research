# WMFIX data need — 2026-10-31 (owner decision material)

## Question
Can the 12 registered FIT worlds (~24 paired contrasts) support the 1473-parameter
merge-response head as a *learned* arm-conditional correction?

## Supply
- Supervision: 12 FIT worlds x 2 arms (ours_c1, coopoff_c1) = 24 paired contrasts,
  one t300 start each, 600 s tapes, MAPPO excluded (lane-head mismatch, WMREQUAL3).
- Schedule: <=120 updates at batch 1 -> each contrast seen 2-3 times (review plausibility).
- Signal: directional paired error (J300 r -0.235, sign .556), upstream covariance 96%.

## Verdict: NO for the head as primary route; YES for everything else
- The entry-path fix (exact conservation + learned admission + zero head) needs NO new data.
- The zero-head retarget check may remove the head's target entirely; if the residual
  vanishes, no collection is justified.
- If the residual survives, 24 contrasts are under-determined for 1,473 params over
  4 broad bases (sigma ~907 m): the review's most-likely outcome is a common-mode level
  shift and "best = step0". A bounded fine-tune remains runnable under the preregistered
  three-way attribution, but the owner should not expect a learned arm-conditional
  response from this corpus alone.

## If the owner wants the head learnable (pick one or both)
1. REUSE FIRST, zero sim cost: enumerate remaining FIT-role worlds in
   /root/autodl-tmp/c4/e3data (the corpus recorded ~156 fit worlds; 12 are allowlisted)
   that already carry native/ours_c1/coopoff_c1 t300 paired tapes; register an allowlist
   amendment. Each usable world adds 2 contrasts.
2. NEW COLLECTION (only for worlds lacking arm tapes): 36 additional TRAIN fit worlds
   (seeds clear of 964000-964199, 962xxx, 977xxx and every validation/eval block)
   x 3 tapes (native + ours_c1 + coopoff_c1) x (300 s warmup + 600 s window)
   = 97,200 sim-s (~100k sim-s budget line; ~2.5x contrasts, 24 -> ~60+reuse).
3. CHEAPER ALTERNATIVE, no new data: shrink and focus the head (bases concentrated on
   the merge/ramp-entrance region per review item 5) so fewer parameters face the same
   24 contrasts with higher resolution where discharge forms.

Recommendation: run the zero-head retarget check first; only then decide between 1+2
and 3. No collection should be registered before that result exists.
