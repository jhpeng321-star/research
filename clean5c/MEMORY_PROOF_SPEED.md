# c4/speed G4 memory proof (8 one-job G4 workers)

This amends `runs/active/C4-V7FT-20261017/clean5bf8s/MEMORY_PROOF.md` for the G4 phase only. Every other phase is unchanged.

cpuZ/cpuW target: the worst-case resident total stays at or below 165 GB.

| Phase | Resident (GB) | Workers | Worst case (GB) | Floor reachable? |
|---|---|---|---|---|
| G4 (both parents resident) | B 86 + R 46 + 9 = 141 | 8 one-job workers x g, g = 2.6 measured (F8R 8/8 jobs) | 141 + 8 x 2.6 = **161.8** <= 165 | Floor 20 GB. With no job running, 184 - 141 = 43 >= 20. Workers exit after each job, so no idle memory is retained. |

- The 8-worker bound holds for a per-worker peak of g <= 24 / 8 = **3.0 GB**. That is 1.15x the measured 2.6 GB. With 4 workers (F8S) the margin was 6.0 GB.
- If a G4 job peaks at 6 GB, 8 concurrent jobs would reach 189 GB. Only 4 workers at 6 GB fit inside 165 GB.
- The guard: a job is admitted only while MemAvailable >= 20 GB. So the used total at each admission is at most 164 GB, plus the growth of the jobs already running.
- The deadlock detector and memwatch stay on (`runner.admission`).
- Recommendation: use 8 workers only with the measured 2.6 GB profile. Set `g4_workers` (prepare_c / PREPARE_INPUTS `g4_workers`) to 4 if a worker's G4 peak exceeds 3.0 GB.

Unmeasured, disclosed: the G4 per-job peak under the clean5c bank templates. Its program sessions may enlarge branch states.
