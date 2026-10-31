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
