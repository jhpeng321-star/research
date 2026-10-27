# S-LONGHZN-1: WM1 passes through 90 seconds

**90 s is the longest passing tested horizon**, unchanged from WM0/base r16; 120/180/300 s fail. Threshold **0.868421**, unchanged. At 90 s WM1's lower CI is 0.851: PASS uses the registered **point estimate**, not a confidence-bound guarantee.

Same pairs300 TRAIN panel: 96 groups/54 worlds, 192 paired differences, 190 nonzero-reference signs at every horizon; original seeds and 2,000-draw group bootstrap. CPU-only, ≤16 inference workers/box on cpuP/cpuQ; zero SUMO/training.

Each line gives **WM0 → WM1 → base r16**, as **sign agreement [95% CI]; Pearson r**:

- **60 s:** .958 [.926,.984]; .753 → .963 [.935,.989]; .774 → .953 [.920,.979]; .774.
- **90 s:** .900 [.844,.947]; .572 → .905 [.851,.953]; .638 → .889 [.832,.938]; .647.
- **120 s:** .763 [.684,.839]; .369 → .779 [.702,.849]; .479 → .753 [.674,.828]; .479.
- **180 s:** .626 [.542,.703]; .305 → .663 [.580,.737]; .456 → .626 [.542,.705]; .452.
- **300 s:** .605 [.521,.693]; .351 → .653 [.573,.734]; .597 → .616 [.532,.702]; .607.

WM1−WM0 sign gains, 60→300 s: **+0.53/+0.53/+1.58/+3.68/+4.74 percentage points**; WM1−r16: **+1.05/+1.58/+2.63/+3.68/+3.68**. Paired 300 s gain versus WM0: +4.74 [1.05,8.95] points.

Residual **e=dJ_WM−dJ_recorded**, mean/centered SD (veh-s), same model order:
- **120 s:** +341/341 → **+304/303** → +315/298.
- **300 s:** +577/605 → **+501/508** → +522/504.

WM1 bias CIs: [249,364]/[408,600]; RMSE 429/713. Bias² accounts for **50.1%/49.3% of MSE**: systematic overpricing of action-minus-native and residual dispersion are equally important. Dispersion includes heterogeneous model error, not identified stochastic noise.

WM1 largely restores base-r16 Pearson accuracy without extending range. This recorded-command, t0-HDV TRAIN gate does not establish closed-loop reliability. Full Pearson CIs, paired changes, hashes and reproduction: [evidence](tools/longhzn/evidence/README.md).
