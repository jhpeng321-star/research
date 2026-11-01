# S-RB2-CAL development receipt

Internal development evidence only. Full-world CIs unavailable: demand × penetration cells are unreplicated.

## Learned physical parameters

- Posterior retention: 1.
- Gamma-rate variance multiplier: 1.6672352.
- Entrant concentration: 0.0 (null means fixed-population infinity).
- Count NLL, whole-world-excluded scalar fits: 3.746607 → 3.745985.

The population itself was fitted on FIT32; scalar crossfits are not independent population validation.

## Development traffic outcomes

### physical

**120 s**
- delay: CRPS 739.36 veh-s; 50/80/90% coverage 25.0/53.3/60.0%.
- paired_delay: CRPS 152.34 veh-s; 50/80/90% coverage 28.3/53.3/59.2%.
- preference: Brier 0.23841, n=120; forecast=0.5822916666666667, observed=0.6333333333333333.
- selected_preference: Brier 0.24592, n=23; forecast=0.9456521739130435, observed=0.6956521739130435.

**300 s**
- delay: CRPS 2841.42 veh-s; 50/80/90% coverage 31.7/55.0/73.3%.
- paired_delay: CRPS 812.34 veh-s; 50/80/90% coverage 29.2/49.2/60.0%.
- preference: Brier 0.26875, n=120; forecast=0.5708333333333333, observed=0.48333333333333334.
- selected_preference: Brier 0.14183, n=13; forecast=0.9423076923076923, observed=0.8461538461538461.

**600 s**
- delay: CRPS 10646.49 veh-s; 50/80/90% coverage 8.3/25.0/53.3%.
- paired_delay: CRPS 2441.76 veh-s; 50/80/90% coverage 20.8/35.0/45.0%.
- preference: Brier 0.24245, n=120; forecast=0.5625, observed=0.5083333333333333.
- selected_preference: Brier 0.19466, n=24; forecast=0.9427083333333334, observed=0.7916666666666666.

### diagnostic_oof_scale

NOT a physical candidate: mean-preserving outcome-scale limitation experiment, scalar fitted excluding entire target world.

**120 s**
- delay: CRPS 747.57 veh-s; 50/80/90% coverage 45.0/66.7/83.3%.
- paired_delay: CRPS 148.91 veh-s; 50/80/90% coverage 39.2/64.2/71.7%.
- preference: Brier 0.20924, n=120; forecast=0.584375, observed=0.6333333333333333.
- selected_preference: Brier 0.14773, n=11; forecast=0.9090909090909091, observed=0.8181818181818182.

**300 s**
- delay: CRPS 2862.56 veh-s; 50/80/90% coverage 43.3/66.7/81.7%.
- paired_delay: CRPS 805.28 veh-s; 50/80/90% coverage 36.7/58.3/65.8%.
- preference: Brier 0.26615, n=120; forecast=0.5604166666666667, observed=0.48333333333333334.
- selected_preference: Brier 0.10938, n=10; forecast=0.925, observed=0.9.

**600 s**
- delay: CRPS 10017.41 veh-s; 50/80/90% coverage 36.7/78.3/85.0%.
- paired_delay: CRPS 2329.42 veh-s; 50/80/90% coverage 32.5/57.5/68.3%.
- preference: Brier 0.23802, n=120; forecast=0.5604166666666667, observed=0.5083333333333333.
- selected_preference: Brier 0.16699, n=16; forecast=0.9140625, observed=0.8125.

Strict paired ties are non-wins. Selected-call sets differ across methods.
Reliability diagrams and exact decile support are adjacent to each raw outcome RESULTS.json.
Reserved8 outcomes were not scored or used for parameter selection. No outcome-coverage qualification is implied.
