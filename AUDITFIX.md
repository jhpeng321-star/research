# Imported S-AUDITFIX six-row preset

Only the six-row preset and its parity test from `24928e9d` are imported here.
The other S-AUDITFIX repairs are not part of this branch.

Select `configs/c4_policy14_c4prep_sixrow.toml` explicitly, or copy its
counterfactual overrides into the resolved benchmark TOML. For the registered
weights, `8.9 D_i + 8.9 (D_G - sum D_i) + 9.9 D_G = 18.8 D_G`.
All/team terms remain unchanged. Three zero-coefficient individual rows are
removed for a three-member group (9 rows becomes 6).

The imported mean and q10/K8 tests compare loss and full actor gradients with
tolerances; they do not establish bitwise equivalence of a real policy update.
Nontrivial real-credit multipliers must not be applied to the rewritten weights
and assumed equivalent to applying them to the original parameterization.
