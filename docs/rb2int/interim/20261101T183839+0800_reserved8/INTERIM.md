## INTERIM v5 versus v4 — 20261101T183839+0800 (reserved8)

**8 matched complete K8 worlds; all five arms.** This is an interim diagnostic, not model selection or switching authorization.
Worlds: 969432, 969433, 969434, 969435, 969436, 969437, 969438, 969439.
CIs use10,000 paired fixed-cell whole-world resamples; delay/HDV/accuracy condition on stored K8. Rate/effect means include paired future resampling. Each metric family is simultaneous across its displayed horizons/models; this interim two-model family is separate from the final registered six-contrast family. No sequential/repeated-look confidence claim.
Strata: [{'demand': 1900, 'penetration': 0.05, 'worlds': 1}, {'demand': 1900, 'penetration': 0.4, 'worlds': 2}, {'demand': 2400, 'penetration': 0.4, 'worlds': 1}, {'demand': 2700, 'penetration': 0.05, 'worlds': 1}, {'demand': 2700, 'penetration': 0.2, 'worlds': 1}, {'demand': 2700, 'penetration': 0.4, 'worlds': 2}].

### Delay, breakdown and HDV60
- FULL-v4 delay CRPS120/300/600=1,105/3,777/13,105veh-s; mean bias=1,263/4,451/14,648veh-s.
- FULL-v5 delay CRPS120/300/600=1,274/4,887/16,101veh-s; mean bias=1,536/6,136/20,502veh-s.
- 120s v5−v4 delay CRPS: 169.0; CI NA (singleton)veh-s; breakdown absolute rate-distance: 8.44; CI NA (singleton)pp.
- 300s v5−v4 delay CRPS: 1,110.3; CI NA (singleton)veh-s; breakdown absolute rate-distance: 2.81; CI NA (singleton)pp.
- 600s v5−v4 delay CRPS: 2,996.2; CI NA (singleton)veh-s; breakdown absolute rate-distance: -0.94; CI NA (singleton)pp.
- v5−v4 HDV60 mean-world-median position-error difference: 1.9; CI NA (singleton)m.

### Controller-effect signs and credibility
- FULL-v4/120s:0.8calls n=17, observed win 52.9; CI NA (singleton)% vs forecast 92.6; CI NA (singleton)%; range coverage 45.0; CI NA (singleton)%; all10pair direction accuracy 56.2; CI NA (singleton)%.
- FULL-v4/300s:0.8calls n=18, observed win 55.6; CI NA (singleton)% vs forecast 92.4; CI NA (singleton)%; range coverage 65.0; CI NA (singleton)%; all10pair direction accuracy 60.0; CI NA (singleton)%.
- FULL-v4/600s:0.8calls n=20, observed win 70.0; CI NA (singleton)% vs forecast 93.1; CI NA (singleton)%; range coverage 57.5; CI NA (singleton)%; all10pair direction accuracy 65.0; CI NA (singleton)%.
- FULL-v4 ctrl0−native expected delay effect120/300/600: -32.5; CI NA (singleton) / -212.7; CI NA (singleton) / -49.5; CI NA (singleton)veh-s; recorded -92.4; CI NA (singleton) / -1,241.8; CI NA (singleton) / -2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 mappo−native expected delay effect120/300/600: -12.8; CI NA (singleton) / -316.7; CI NA (singleton) / -1,246.4; CI NA (singleton)veh-s; recorded 57.0; CI NA (singleton) / -661.8; CI NA (singleton) / -2,013.6; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 pi−native expected delay effect120/300/600: -64.3; CI NA (singleton) / -427.8; CI NA (singleton) / -1,674.2; CI NA (singleton)veh-s; recorded 10.9; CI NA (singleton) / -572.7; CI NA (singleton) / -1,130.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 circles−native expected delay effect120/300/600: 349.2; CI NA (singleton) / 1,417.7; CI NA (singleton) / 4,583.1; CI NA (singleton)veh-s; recorded 305.5; CI NA (singleton) / 717.0; CI NA (singleton) / 3,746.1; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 native−ctrl0 expected delay effect120/300/600: 32.5; CI NA (singleton) / 212.7; CI NA (singleton) / 49.5; CI NA (singleton)veh-s; recorded 92.4; CI NA (singleton) / 1,241.8; CI NA (singleton) / 2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 mappo−ctrl0 expected delay effect120/300/600: 19.8; CI NA (singleton) / -104.0; CI NA (singleton) / -1,196.9; CI NA (singleton)veh-s; recorded 149.5; CI NA (singleton) / 580.0; CI NA (singleton) / 539.0; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 pi−ctrl0 expected delay effect120/300/600: -31.7; CI NA (singleton) / -215.1; CI NA (singleton) / -1,624.7; CI NA (singleton)veh-s; recorded 103.3; CI NA (singleton) / 669.1; CI NA (singleton) / 1,421.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 circles−ctrl0 expected delay effect120/300/600: 381.8; CI NA (singleton) / 1,630.4; CI NA (singleton) / 4,632.6; CI NA (singleton)veh-s; recorded 397.9; CI NA (singleton) / 1,958.8; CI NA (singleton) / 6,298.7; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5/120s:0.8calls n=14, observed win 71.4; CI NA (singleton)% vs forecast 95.5; CI NA (singleton)%; range coverage 37.5; CI NA (singleton)%; all10pair direction accuracy 57.5; CI NA (singleton)%.
- FULL-v5/300s:0.8calls n=10, observed win 70.0; CI NA (singleton)% vs forecast 93.8; CI NA (singleton)%; range coverage 52.5; CI NA (singleton)%; all10pair direction accuracy 56.2; CI NA (singleton)%.
- FULL-v5/600s:0.8calls n=13, observed win 69.2; CI NA (singleton)% vs forecast 97.1; CI NA (singleton)%; range coverage 62.5; CI NA (singleton)%; all10pair direction accuracy 68.7; CI NA (singleton)%.
- FULL-v5 ctrl0−native expected delay effect120/300/600: 44.3; CI NA (singleton) / 350.8; CI NA (singleton) / 787.6; CI NA (singleton)veh-s; recorded -92.4; CI NA (singleton) / -1,241.8; CI NA (singleton) / -2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 mappo−native expected delay effect120/300/600: 29.6; CI NA (singleton) / 84.7; CI NA (singleton) / -252.2; CI NA (singleton)veh-s; recorded 57.0; CI NA (singleton) / -661.8; CI NA (singleton) / -2,013.6; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 pi−native expected delay effect120/300/600: -30.8; CI NA (singleton) / 67.9; CI NA (singleton) / 67.7; CI NA (singleton)veh-s; recorded 10.9; CI NA (singleton) / -572.7; CI NA (singleton) / -1,130.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 circles−native expected delay effect120/300/600: 354.4; CI NA (singleton) / 1,832.9; CI NA (singleton) / 6,398.6; CI NA (singleton)veh-s; recorded 305.5; CI NA (singleton) / 717.0; CI NA (singleton) / 3,746.1; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 native−ctrl0 expected delay effect120/300/600: -44.3; CI NA (singleton) / -350.8; CI NA (singleton) / -787.6; CI NA (singleton)veh-s; recorded 92.4; CI NA (singleton) / 1,241.8; CI NA (singleton) / 2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 mappo−ctrl0 expected delay effect120/300/600: -14.7; CI NA (singleton) / -266.1; CI NA (singleton) / -1,039.8; CI NA (singleton)veh-s; recorded 149.5; CI NA (singleton) / 580.0; CI NA (singleton) / 539.0; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 pi−ctrl0 expected delay effect120/300/600: -75.1; CI NA (singleton) / -282.8; CI NA (singleton) / -719.9; CI NA (singleton)veh-s; recorded 103.3; CI NA (singleton) / 669.1; CI NA (singleton) / 1,421.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 circles−ctrl0 expected delay effect120/300/600: 310.1; CI NA (singleton) / 1,482.1; CI NA (singleton) / 5,611.0; CI NA (singleton)veh-s; recorded 397.9; CI NA (singleton) / 1,958.8; CI NA (singleton) / 6,298.7; CI NA (singleton)veh-s. Negative means lower delay.

### Short-horizon decision quality:30/60/90s
Exact total-delay accuracy is **partially identified**: the recorded data saved total/source clocks only at120/300/600s. Reconstructed road clocks match recorded120s road delay; unknown short source waiting is bounded between0 and each arm's recorded120s source waiting. The intervals below bound all-pair team-effect accuracy without inventing a source trajectory. Road-only accuracy is a separate diagnostic.
These are five recorded controller options, not unrecorded cooperation membership/group interventions. Cooperation/planner short-horizon qualification remains NOT ESTABLISHED.
- FULL-v4/30s: team direction accuracy identification lower 5.0; CI NA (singleton)%, upper 100.0; CI NA (singleton)%; resolved-pair fraction 5.0; CI NA (singleton)%; resolved-only accuracy 100.0; CI NA (singleton)%; road-only direction accuracy 66.2; CI NA (singleton)%.
- FULL-v4/60s: team direction accuracy identification lower 12.5; CI NA (singleton)%, upper 97.5; CI NA (singleton)%; resolved-pair fraction 15.0; CI NA (singleton)%; resolved-only accuracy 83.3; CI NA (singleton)%; road-only direction accuracy 60.0; CI NA (singleton)%.
- FULL-v4/90s: team direction accuracy identification lower 13.8; CI NA (singleton)%, upper 96.2; CI NA (singleton)%; resolved-pair fraction 17.5; CI NA (singleton)%; resolved-only accuracy 78.6; CI NA (singleton)%; road-only direction accuracy 61.3; CI NA (singleton)%.
- FULL-v5/30s: team direction accuracy identification lower 5.0; CI NA (singleton)%, upper 100.0; CI NA (singleton)%; resolved-pair fraction 5.0; CI NA (singleton)%; resolved-only accuracy 100.0; CI NA (singleton)%; road-only direction accuracy 67.5; CI NA (singleton)%.
- FULL-v5/60s: team direction accuracy identification lower 12.5; CI NA (singleton)%, upper 97.5; CI NA (singleton)%; resolved-pair fraction 15.0; CI NA (singleton)%; resolved-only accuracy 83.3; CI NA (singleton)%; road-only direction accuracy 65.0; CI NA (singleton)%.
- FULL-v5/90s: team direction accuracy identification lower 16.2; CI NA (singleton)%, upper 98.8; CI NA (singleton)%; resolved-pair fraction 17.5; CI NA (singleton)%; resolved-only accuracy 92.9; CI NA (singleton)%; road-only direction accuracy 70.0; CI NA (singleton)%.

Strict paired P(beats incumbent)>=0.8 plus lower expected delay; ties are non-wins. K8 min–max range is not a90% interval (continuous exchangeability reference7/9). Zero-selected worlds remain in resampling. Full gate/switching: NOT ESTABLISHED at this interim.
