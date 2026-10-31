# One audited infrastructure retry, not a repeated scientific test

- First short dispatcher29400 paused/drained all its own short jobs, then exited.
  It preserved84combined completed tasks; V/D/E and efficacy evaluator unopened.
- s969433_t540_k6 SSH returned255. Entire original log is exactly one line:
  `Connection closed by 172.19.0.17 port 11000`. No Python START/exception.
- Before the explicit MAIN21:12 once-only retry: remote output absent; gpuA GPU
  has zero compute PIDs; complete process metadata shows only original owned
  stdin CPU workers251940/251994, no unknown stdin worker. PID29400 absent.
- retry_infra audits these conditions, preserves old claim/log/error, and uses
  new exclusive per-key infra_retry_once receipts. Exact same short_v1 worker,
  frozen runtime/CRN/input paths; no fitting or outcome scoring.
- Retry completed29.45s CUDA. Receipt/claim/log under
  _out/wmvalue_short_v1/infra_retry_once/s969433_t540_k6; result copied once to
  the original short tasks directory. Old error retained; no overwritten output.
- Original s969432_t300_k0 was alive at22:06 metadata and was NOT retried.
  It subsequently completed8897.033s on original PID251940; its t900/1200step
  output was collected. Separate retry authorization remains unconsumed.
- launch --resume requires proof the original dispatcher is no longer alive,
  preserves LAUNCH/DISPATCH logs, writes exclusive RESUME receipt/log, and reuses
  existing claims without redispatching them. No peer PID signals or new model.
- Resume verified at22:10, exact local PID43404; durable RESUME.json/RESUME.log
  in the same short root. Initial collection87complete/5oldCPU active; then
 93complete with7shortCPU+3GPU workers and5oldCPU workers, no new errors.
  Worker/runtime/evaluator source hashes are identical to the initial short launch.
- At22:25 the combined prefix count passed160/320; both gpuA/gpuR admitted
  short CUDA tasks only when peer compute PIDs were absent. No outcome scoring.
  Read-only monitor now reports RESUME's current PID, combined old+short coverage,
  and distinguishes preserved historical errors from unresolved missing outputs.
