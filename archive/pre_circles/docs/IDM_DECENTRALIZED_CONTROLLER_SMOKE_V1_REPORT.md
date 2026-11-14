# Decentralized IDM Controller SUMO Smoke V1

Status: software integration passed; efficacy not evaluated.

Command:

```powershell
C:\Users\jpeng\Miniconda3\envs\rl_driving\python.exe `
  -m tools.smoke_idm_decentralized_controller
```

Artifact: `runs/eval/idm_decentralized_controller_smoke_v1.json`.

The smoke used IDM demand 2200, development seed 4107, 20% CAV penetration,
and 300 post-warmup controller steps. Every admissible CAV requested nominal
gain mode 2; this was an integration test rather than an authority comparison.

Observed results:

- `Discrete(5)` action space and five mask bits;
- 144 candidate-admission records;
- 62 rank-two-or-later candidate records;
- 55 admissible records;
- eight selected 20-step options;
- 160 controlled CAV-seconds;
- maximum continuous option duration 20 steps;
- one enforced cross-lane spacing conflict;
- zero all-zero masks, illegal actions, and commands without a legal bid;
- all commands used the requested nominal mode;
- 89 geometric candidates rejected for less than 16 s predicted dose.

This establishes that the implemented candidate enumeration, fallback
exposure, local action mask, contention, latch, and command plumbing run in real
SUMO. One inspected development seed cannot establish safety rates, service
bounds, placement benefit, mode benefit, generalization, or training readiness.
The next authorized experiment is a small paired authority screen, not PPO.
