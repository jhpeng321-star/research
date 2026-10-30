# Retained host-local CPU runtime benchmark
Runtime `gpuA:/root/autodl-tmp/c4/candB_rankcheck_32bd0f90_cpu_20261030`.
Source32bd0f90 differs from primary05f7bfba only in report tooling and backend metadata.
Exact same WM/objective/mix/K8/starts/inputs; deviceCPU, CUDA hidden, two threads, nice19.
The first CPU future exceeded230s versus primary CUDA116s. Backend selection used
runtime only, before any CPU arm result. Owned worker957380/start873765086 was
terminated after verifying its recorded incarnation. Original files retained.
Primary remains dense-slot CUDA05f7bfba at allocator.40, full16-start panel.
Empty-slot compaction was inspected/tested (8 existing tests passed) but NOT enabled:
its float reassociation would add an unnecessary execution-mode difference.
