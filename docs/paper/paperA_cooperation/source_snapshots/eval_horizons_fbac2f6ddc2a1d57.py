r"""S-COOPEVAL: evaluation + figure pipeline for the S-SUMOCF cooperation paper.

Read-only over a mirrored S-SUMOCF run tree; no SUMO, no training, no new seeds.
Registered conventions: paired world bootstrap, 2000 draws, seed 0, 95 % percentile
CIs; J = E3 team_delay_vs (fleet + source delay, veh-s, lower is better);
learned-minus-comparator, negative = better; credit D positive = better with
C_G = D_G - sum_i D_i and M_i = D_G - D_(G\i).
"""
VERSION = "1"
DRAWS = 2000
SEED = 0
LEVEL = 0.95
HORIZONS = (120, 300, 600)
