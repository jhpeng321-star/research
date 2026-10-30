# gpuR CIRCLES continuation — pending until large-start worker finishes
Source4d56f2a8, exported plan indices12-15, original K8 first4 CRN futures.
Default-OFF measurement-only pair-bias row chunks8; exact bound objective/weights.
Only dispatch after the own large-start worker exits with its shard_finished receipt,
and total gpuR preflight including worst.40 reservation plus512MiB is <=22000MiB.
No parallel new scorer or gpuA dispatch. Each complete arm is retained as written.
