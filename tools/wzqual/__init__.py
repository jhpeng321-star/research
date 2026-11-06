"""WM-D: world-model fidelity and qualification suite for the wz3to2 lane-drop map.

Modules (all CPU, numpy only; the WM itself is reached through a backend adapter):

* :mod:`frame`     wz3to2 geometry in the WM model frame (s, WM lane), zones and detectors
* :mod:`traj`      canonical trajectory table + loaders (WM-D prefix tapes, c4 episodes, frames)
* :mod:`traffic`   traffic series at the lane drop: team delay, zone speed, queue, throughput
* :mod:`breakdown` two-regime breakdown rule (fitted, frozen) and breakdown events / onset
* :mod:`errors`    car-level and traffic-level multi-step prediction errors
* :mod:`distrib`   distributional comparison WM vs SUMO (breakdown rate, onset, queue, flow, CRPS)
* :mod:`labels`    SUMO-labelled cooperation moves (N3 / N8 / PLANGEN / LOWCAV) as candidates
* :mod:`moves`     backend-agnostic executor of the labelled move rules (hold/pulse/tau/lane roles)
* :mod:`ranking`   action-response: rank correlation, precision, precision@k, regret, bootstrap
* :mod:`backends`  backend protocol, keep-speed baseline, fake backend for tests
* :mod:`gate`      the pre-registered pass/fail gate (WM_QUAL_GATE.md / gate_spec.json)
* :mod:`run_qual`  CLI: starts -> backend rollouts -> metrics -> gate verdict -> report
"""
