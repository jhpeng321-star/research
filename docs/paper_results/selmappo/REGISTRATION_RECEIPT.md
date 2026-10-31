# Exact fleet budget registered before simulation

Shared ledger: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`.
The single `S-SELMAPPO DESIGN REGISTERED BEFORE COMPUTE` line binds:

- Design commit `7e4def77`, build `a5dafd3b`.
- Design SHA256 `992c05601f03d1d9caad0421b041dbeaa05fa8621ae7e3020be42e3c9f5b41d0`.
- Exact fleet cap **7,012,800sim-s** =6,996,600production +1,800mechanics +14,400evaluation;
  no per-CAV allocation here, no GPU allocation, no retries/refunds.
- Fleet TRAIN18000000..18007999, exact production/smoke split in design; all CPU/exposure gates.
- MAIN23:42 advance owner approval and23:46 scope amendment already apply.

Post-append verification: `registered(...)` PASS and registry interval audit PASS,
registry snapshot SHA256 `4c6ef181d3828e58ad6729e0e42a57717500c89cb77300005e2809f059a68b0f`.
No SUMO was run before that append.

S-WALLSW handoff subsequently arrived (`07fb123e`); actual published library bound and
real CPU integration PASS at immutable stage `build_bc4c3aae72073204`, source binding
`bc4c3aae72073204cf046d800766d4fcc5fd9610a29ea099c3c40cfc195dce14`.
Full staged CPU receipt is `STAGE_bc4c3aae72073204.json` alongside this receipt.
Training launch reuses the same approved cap; final evaluation must still use the
registered16world S-C1SWITCH2 panel and must not duplicate fleet episodes across tasks.

## First attempt and stop reconciliation
Own dispatch PID284174 exited before physics due to a captured frozen guard alias.
The shared ledger now also contains `S-SELMAPPO FIRST DISPATCH MECHANICAL STOP / CPU REPAIR READY`:
900reserved/0actual retained,29CPU tests +actual frozen-alias integration PASS,
repair stage914fecba3ffdf4cc staged only, same-cap continuation proposal awaiting MAIN.
Post-stop exact design-registration and zero-conflict interval checks PASS;
registry snapshot SHA256 `ccd1d0814eb524add9df665fc53d4dcae75db4293582c9ac791fbb2ee71da8d5`.

## MAIN acknowledgement and terminal stop
MAIN00:29 explicitly acknowledged the first stop under the same cap and authorized
only the unused second mechanics episode, followed by production. Commit `c082b233`
implemented one-use continuation, inherited the exact first charge, and refused replay.
Dispatch ownPID357398 reached900actual on the second mechanics episode, then failed
terminal clock/cleanup bookkeeping because the frozen simulator had already closed api.
Shared ledger now contains `S-SELMAPPO SECOND MECHANICS TERMINAL STOP / MAIN ACK NEEDED`:
1,800charged /900actual, zero production/checkpoints, both mechanics slots consumed.
CPU31/31 and frozen integration PASS at CPU-only stage68394a9568862929; no redispatch.
Exact design-registration and interval audit PASS after append, registry snapshot SHA256
`0c23689663962b83f8a95ad2e0e8ecc262a9d4c26868ba3355545fa7ad795cff`.
Evaluation ownership follows C1SWITCH2's registered line: leave our14,400 reserved sim-s
unstarted and reuse peer-owned16world receipts. Production-only continuation needs
MAIN acknowledgement of this new terminal failure; nothing automatically launches.
