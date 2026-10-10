"""WM-N1-DSPEED (A9): make the direct-speed ``trajectory`` runs trainable.

Self-contained pipeline that turns the read-only
``E:/research-dense/runs/active/WM-N1-DSPEED-20261004/{box,laptop}/runs``
trajectory runs into:

* per-run v0.4a label parquets (:mod:`dspeed.labels`);
* one packed memmap corpus in the frozen ``wm_n1_pack/1`` v3 schema
  (:mod:`dspeed.build_pack`), with small per-run alignment receipts;
* a schema/3-style TRAIN outcome overlay sidecar keyed by full item
  identity (:mod:`dspeed.overlay`).

All outputs live under the workspace ``_out/`` tree; the campaign runs and
the frozen pack are never written.  TRAIN worlds only.
"""
