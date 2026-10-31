# CPU throughput canary — operationally stopped, no efficacy verdict

Source95e81e79493a2817df499b51b899d81d0c272f13; immutable gpuA CPU-only root
`/root/rbeval20261031/int/WM-PHYS-FULL-v2-cpu`.
One proposed five-arm600s canary launched Unix1793473598.1043942, own PID356826,
boot/start identity in CANARY_PROCESS.json. At03:16 it had585CPU seconds,
approximately1.5GiB RSS and no completed task. The exact owned incarnation was
cooperatively stopped then SIGTERM'd; CANARY_STOPPED.json verifies absent/0tasks
at Unix1793474380.1550813 (~782wall seconds). No efficacy scores were inspected.

This is a throughput-route rejection, NOT a measured physics/accuracy failure.
No automatic retry/panel continuation. All source/logs/partial work remain on
gpuA; original GPU FULL-v1/A2/B2 jobs are untouched. No SUMO or training.
