# REGISTRATION_WM_N1 — AMENDMENT 1: training ORDER (member-1-first) and dedicated GPU box

**Status:** owner decision relayed by main discussion, 2026-09-30.

- Amends: `REGISTRATION_WM_N1.md`, LF-sha256
  `666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246`. That
  file is NOT edited.
- Changes the training ORDER and the compute location only. The §4 targets,
  the chunk-5 TEST (`REGISTRATION_CHUNK5_TEST.md`, LF-sha256
  `f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304c38214b31e`), the
  three variants (N/T/C), the losses, the architecture and the CAL-only
  calibration are UNCHANGED.

## A1. Member-1-first

1. After the 1-h timing gate and the owner's approval file, train ensemble
   member 1 ALONE to completion, giving an early working model.
2. Then train members 2–5, with the same config and independent seeds
   recorded before training.

## A2. Use of the early (1-member) model

- Only internal checks on TRAIN (OOF/in-sample) and CAL are allowed, for
  example loss curves, calibration diagnostics and a planner integration
  smoke.
- It supports NO selection or superiority claim.
- CAL use for the early model is diagnostic only. It must not change the
  architecture, the hyperparameters or the variant rule. The registered
  CAL-only calibration and stacking are fitted once, on the full 5-member
  ensemble.
- Chunk 5 stays SEALED until the single final test. That test uses the full
  5-member ensemble and the frozen PLANNER_VARIANT.json. The early model is
  never run on chunk 5.

## A3. Compute

- A dedicated second GPU box runs WM-N1 only, independent of the recon GPU
  box. It uses the same env spec and the same guard and sync.
- Its queue is: verify upload → memmap verify → 1-h timing gate STOP → owner
  approval file → member 1 → members 2–5.
- The design §7 "shared box" text is superseded for location only.
