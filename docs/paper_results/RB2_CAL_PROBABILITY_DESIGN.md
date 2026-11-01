# Development-only strict-win probability calibration (before fitting)

Motivation: calibrated latent draws still require a measured decision-confidence
check. Add a separate, explicit reporting/decision map; never change trajectory
costs, event labels, physical futures, driver means or the 0.8 switching rule.

For every ordered strict-win probability, preserve its predicted tie mass `t`:
`p_cal = lambda*p_raw + (1-lambda)*(1-t)/2`, `lambda in [0,1]`.
This keeps reverse probabilities summing to `1-t`, unlike unrelated sigmoid fits.
Learn one scalar per horizon by minimum Brier on all ten unordered arm pairs
on exactly development worlds969420–969431. Endpoints are exact and optimization
is quadratic. No arm ID, traffic threshold or neural model is fitted.

Evaluate each world using lambda fitted excluding that WHOLE world. Report all
pair Brier/reliability and the actual selected-call Brier/reliability, support,
switch fraction and recorded saving after applying the unchanged mean/0.8 rule.
No-switch worlds are not wins or probability cases. Empty selected sets are
explicitly unqualified; eliminating calls cannot establish useful 0.8-call skill.

Bind frozen maps to the exact producing model-contract fingerprint, which includes
all source/config/artifact hashes. Refuse applying a map to a changed model.
Integrator must refit on dev after combining physics changes. The learned
parameter and OOF diagnostics remain internal development evidence, not final
comparison evidence. No reserved outcome task is read, and sparse-cell CIs stay NA.
