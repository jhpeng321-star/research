"""Preserved historical seed exclusions; absence here is not proof of freshness.

Partitions extracted from idm/contract.py and idm/damping_campaign.py during
CIRCLES adoption. Reserved acceptance blocks stay sealed. New prospective
registration must audit archived evidence; it must not infer freshness from
this list alone. No simulator data are read by this module.
"""
import hashlib
import json
from pathlib import Path

NATIVE_DAMPING_TRAINING_SEEDS = tuple(range(910001, 910017))
NATIVE_DAMPING_DEVELOPMENT_SEEDS = tuple(range(911001, 911009))
NATIVE_DAMPING_ACCEPTANCE_SEEDS = tuple(range(912001, 912033))

DEVELOPMENT_SEEDS = tuple(range(4101, 4121))
ANALYTICAL_PILOT_SEEDS = tuple(range(4101, 4107))
# Added only after the 4107--4120 feature-family diagnosis was complete.
# These remain model-development evidence and never enter checkpoint acceptance.
CONTROLLER_FEATURE_VALIDATION_SEEDS = tuple(range(4121, 4133))
# 2026-09-10, registered (OE-17 duration-selection pilot directive):
# audited development-evaluation block for the prospective arm of the
# duration-selection study. Audit (OE-17 registration) found zero prior use in
# any registry, the ledger, runs/active, output, docs or the archive, and no
# sealed overlap. Development evidence only; never prospective/acceptance data.
OE17_DURATION_SELECTION_SEEDS = (4145, 4146, 4147, 4148)
# OE-32 registered fixed-interface observation study, 2026-09-10.
# Metadata exposure audit: output/observation_mode27_v1/design/SEED_AUDIT.json.
# Internal fit/holdout are development evidence, never sealed acceptance.
OE32_OBSERVATION_DEVELOPMENT_SEEDS = tuple(range(4149, 4161))
# OE-34 registered spatial/temporal observation validation, 2026-09-11.
# Metadata audits: output/observation_relational_v1/evidence/SEED_AUDIT_V2.json
# and SEED_AUDIT_SUPPLEMENT.json. Fit/validation are development, not acceptance.
OE34_RELATIONAL_DEVELOPMENT_SEEDS = tuple(range(90001, 90061))

# OE-59 conditional-panel corridor block (920002-920004): fresh identities from the
# documented outcome-independent 92xxx rule; exposure audit receipt sha256
# 5e6d2029d02286002968cf0559f0c82cc71b91aac39bb3214e51ce007e23a845 (16 surfaces,
# identity-aware, 2026-09-14); development role, never sealed acceptance.
OE59_CONDITIONAL_PANEL_SEEDS = tuple(range(920002, 920005))
# OE-62 conditional fresh-seed screen block (920005-920007): fresh identities
# from the documented outcome-independent 92xxx rule FRESH-SEED-OE62/1
# (successor of FRESH-SEED-OE57/1; amendments A1/A2 frozen before the audit);
# exposure audit receipt sha256 4976fb5a5001c589685b1aa6d4b646edd4ee2b7a38152034dd8d4def9aa75c7d (16 surfaces, identity-aware,
# 2026-09-14); baseline lineage: OE-57/OE-60-era receipt sha256
# 5e6d2029d02286002968cf0559f0c82cc71b91aac39bb3214e51ce007e23a845; the two known 920005 token mentions (this file's OE-59 range
# end bound and the OE-61 negative-provenance test literal) are verified NON-use
# under frozen classes N1/N2 of the successor rule; development role, never
# sealed acceptance.
OE62_CONDITIONAL_SCREEN_SEEDS = tuple(range(920005, 920008))
# OE-65 dataset panel block: 36 fresh identities from the documented
# outcome-independent 92xxx rule FRESH-SEED-OE65/1 (successor of
# FRESH-SEED-OE62/1; the balanced six-loading allocation was frozen before
# the audit; amendments A3/A4 recorded after the first finalize); exposure
# audit receipt sha256 aa4b03616f71961787c94e149ac68690b3fffceba7de4d17ad8de7dd74d2be2e
# (16 surfaces, identity-aware, M3b carried, pre-implementation gate passed
# 36/36 with zero blocking occurrences); balanced across the six
# historically-used loadings (six ascending blocks of six identities: first
# four discovery, last two evaluation, loadings in the frozen order 1900,
# 2000, 2100, 2200, 2600, 2700); development role, never sealed acceptance.
OE65_DATASET_SEEDS = tuple(range(920008, 920044))
# 2026-09-09, registered (OE-09 execution directive): audited unrestricted
# development block for the entry-pairs block-2 campaign. Audit found no archived
# assignment, no current-era execution, no sealed overlap and no s-context mention.
# Development evidence only; never prospective/acceptance data.
OE09_DEVELOPMENT_SEEDS = tuple(range(4133, 4145))
VALIDATION_SEEDS = tuple(range(4201, 4241))
DAMPING_ACCEPTANCE_SEEDS = tuple(range(4201, 4221))
DAMPING_CONFIRMATION_SEEDS = tuple(range(4221, 4241))
WALL_DEVELOPMENT_SEEDS = tuple(range(4301, 4321))
WALL_V5_2_PILOT_SEEDS = tuple(range(4301, 4313))
PREDICTOR_DEVELOPMENT_SEEDS = tuple(range(4401, 4521))
PREDICTOR_VALIDATION_SEEDS = tuple(range(4601, 4661))
FINAL_TEST_SEEDS = tuple(range(5001, 5061))
# 2026-09-02: Stage 1a sustained-wall effect-existence development block.
# Never used before this register (grep-audited); excluded from prospective
# evaluation with the other development and validation partitions.
SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS = tuple(range(6001, 6031))
# 2026-09-02: Stage 1b-prep band locator block (native-IDM arm C only): 30
# seeds per lower-demand cell. Never used before this register (grep-audited).
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_1900_SEEDS = tuple(range(6031, 6061))
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS = tuple(range(6061, 6091))
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS = tuple(range(6091, 6121))
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_SEEDS = (
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_1900_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS
)
# 2026-09-03: Stage 1b-prep-5 locator blocks (2400/2600/2700 veh/h, p 0.20,
# ramp 400, long IDM map; 2700 block announced by the amendment).
# Grep-audited before registration (no prior assignment).
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2400_SEEDS = tuple(range(6121, 6151))
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2600_SEEDS = tuple(range(6151, 6181))
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2700_SEEDS = tuple(range(6181, 6211))
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_SEEDS = (
    SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2400_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2600_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2700_SEEDS
)
# 2026-09-03: Stage 1b generation population (demand drawn per seed in
# [2200, 2800] veh/h, ramp 400; p = 0.20 seeds 7001-7240, p = 0.30 seeds
# 7241-7360). The drawn demand is a hidden variable (recorded, never a
# feature). Grep-audited before registration.
STAGE1B_GENERATION_P20_SEEDS = tuple(range(7001, 7241))
STAGE1B_GENERATION_P30_SEEDS = tuple(range(7241, 7361))
STAGE1B_GENERATION_SEEDS = STAGE1B_GENERATION_P20_SEEDS + STAGE1B_GENERATION_P30_SEEDS
# 2026-09-03: Stage 1c release population (plateau-decline demand profile,
# p = 0.20). Grep-audited before registration.
STAGE1C_RELEASE_2200_SEEDS = tuple(range(7401, 7431))
STAGE1C_RELEASE_2400_SEEDS = tuple(range(7431, 7461))
STAGE1C_RELEASE_2600_SEEDS = tuple(range(7461, 7491))
STAGE1C_RELEASE_2700_SEEDS = tuple(range(7491, 7521))
STAGE1C_RELEASE_SEEDS = (STAGE1C_RELEASE_2200_SEEDS + STAGE1C_RELEASE_2400_SEEDS
                         + STAGE1C_RELEASE_2600_SEEDS + STAGE1C_RELEASE_2700_SEEDS)
# 2026-09-05: Stage 2 final-generation population (plateau-decline demand,
# M drawn per seed in [2200, 2800] veh/h from the seed number, ramp 400;
# p = 0.20 seeds 8001-8240, p = 0.30 seeds 8241-8360). The drawn demand is a
# hidden variable (recorded, never a feature). Grep-audited before
# registration.
STAGE2_GENERATION_P20_SEEDS = tuple(range(8001, 8241))
STAGE2_GENERATION_P30_SEEDS = tuple(range(8241, 8361))
STAGE2_GENERATION_SEEDS = STAGE2_GENERATION_P20_SEEDS + STAGE2_GENERATION_P30_SEEDS
# 2026-09-20: CV-20 multi-seed shared-prefix corpus block (controller
# arbitration labels + world-model training evidence, seeds 7001-7012).
# Never used before this register (grep-audited); excluded from prospective
# evaluation with the other development partitions.
CV20_MULTISEED_CORPUS_SEEDS = tuple(range(7001, 7013))
# 2026-09-21: CV-28 fresh-seed validation holdout block (unbiased v2
# calibration; grep-audited; excluded from prospective evaluation).
CV28_VALIDATION_HOLDOUT_SEEDS = tuple(range(7101, 7103))
# 2026-09-21: CV-29 corpus extension block (arbitration training data +
# wall-arm evidence; grep-audited; excluded from prospective evaluation).
CV29_CORPUS_EXTENSION_SEEDS = tuple(range(7301, 7307))
# 2026-09-22: CV-31 controller-coverage extension block (fresh arbitration
# labels: 4 congested m2700 cells targeting the native/PI boundary, 2
# mid-demand m2400 cells for a new regime; grep-audited fresh 2026-09-22,
# never registered before; excluded from prospective evaluation).
CV31_COVERAGE_EXTENSION_SEEDS = tuple(range(7531, 7537))
# 2026-09-25: WM-28 fresh-world validation block (revision v3).
# 80 identities 920044-920123, audited 2026-09-24/25 as provably unused
# (successor of the OE-59/62/65 92xxx blocks) and registered in
# chat; each seed is accepted ONLY with its own planned mainline loading
# from the registered plan (REGISTRATION_V2/V3, plan seed 20260926).
WM28_FRESH_VALIDATION_SEEDS = frozenset(range(920044, 920124))
WM28_SEED_LOADING_MAP = {
    920044: 2700, 920045: 2600, 920046: 2600, 920047: 2700,
    920048: 2000, 920049: 2600, 920050: 2400, 920051: 2600,
    920052: 2700, 920053: 2600, 920054: 2700, 920055: 2600,
    920056: 2000, 920057: 2700, 920058: 2700, 920059: 2400,
    920060: 2600, 920061: 2700, 920062: 2000, 920063: 2000,
    920064: 2600, 920065: 2400, 920066: 2600, 920067: 2700,
    920068: 2600, 920069: 2600, 920070: 2400, 920071: 2600,
    920072: 2700, 920073: 2700, 920074: 2600, 920075: 2700,
    920076: 2700, 920077: 2600, 920078: 2700, 920079: 2700,
    920080: 2400, 920081: 2700, 920082: 2700, 920083: 2600,
    920084: 2600, 920085: 2700, 920086: 2700, 920087: 2700,
    920088: 2700, 920089: 2700, 920090: 2700, 920091: 2700,
    920092: 2600, 920093: 2700, 920094: 2400, 920095: 2000,
    920096: 2700, 920097: 2600, 920098: 2700, 920099: 2600,
    920100: 2600, 920101: 2600, 920102: 2700, 920103: 2400,
    920104: 2700, 920105: 2600, 920106: 2600, 920107: 2600,
    920108: 2700, 920109: 2400, 920110: 2700, 920111: 2700,
    920112: 2600, 920113: 2700, 920114: 2600, 920115: 2700,
    920116: 2700, 920117: 2700, 920118: 2700, 920119: 2700,
    920120: 2600, 920121: 2700, 920122: 2700, 920123: 2700
}
# 2026-09-25: WM-34 clean-retest fresh-world block (registered in chat,
# 2026-09-25). 100 identities 920144-920243: the first contiguous provably
# unused run of the audited pool 920124-920243 (successor of the WM-28 block
# 920044-920123; the pool tail 920124-920143 is left untouched - the WM-28
# ledger documents it as the historical candidate-pool tail, so the WM-34
# audit's contiguity rule conservatively skips it). Exposure audit:
# output/fresh_validation_v2 results_audit_wm34_v3 (WM-34 package). Each
# seed is accepted ONLY with its own planned mainline loading from the
# registered WM-34 plan (WORLD_PLAN.json, plan seed 20260927; 60 worlds at
# 2700, 40 at 2600 veh/h; all 100 TEST).
WM34_FRESH_RETEST_SEEDS = frozenset(range(920144, 920244))
WM34_SEED_LOADING_MAP = {
    920144: 2600, 920145: 2700, 920146: 2600, 920147: 2600,
    920148: 2700, 920149: 2600, 920150: 2700, 920151: 2700,
    920152: 2700, 920153: 2700, 920154: 2600, 920155: 2600,
    920156: 2700, 920157: 2600, 920158: 2700, 920159: 2700,
    920160: 2600, 920161: 2700, 920162: 2600, 920163: 2700,
    920164: 2600, 920165: 2700, 920166: 2700, 920167: 2600,
    920168: 2600, 920169: 2600, 920170: 2600, 920171: 2700,
    920172: 2700, 920173: 2600, 920174: 2600, 920175: 2600,
    920176: 2700, 920177: 2700, 920178: 2700, 920179: 2700,
    920180: 2600, 920181: 2600, 920182: 2700, 920183: 2700,
    920184: 2700, 920185: 2700, 920186: 2600, 920187: 2700,
    920188: 2600, 920189: 2700, 920190: 2700, 920191: 2600,
    920192: 2600, 920193: 2700, 920194: 2600, 920195: 2700,
    920196: 2600, 920197: 2600, 920198: 2700, 920199: 2700,
    920200: 2600, 920201: 2700, 920202: 2700, 920203: 2700,
    920204: 2700, 920205: 2600, 920206: 2700, 920207: 2600,
    920208: 2700, 920209: 2700, 920210: 2600, 920211: 2600,
    920212: 2700, 920213: 2700, 920214: 2700, 920215: 2700,
    920216: 2700, 920217: 2600, 920218: 2700, 920219: 2600,
    920220: 2700, 920221: 2700, 920222: 2600, 920223: 2700,
    920224: 2700, 920225: 2700, 920226: 2700, 920227: 2700,
    920228: 2700, 920229: 2600, 920230: 2700, 920231: 2600,
    920232: 2700, 920233: 2700, 920234: 2600, 920235: 2700,
    920236: 2700, 920237: 2600, 920238: 2600, 920239: 2700,
    920240: 2700, 920241: 2700, 920242: 2600, 920243: 2700,
}
# 2026-09-25: WM-41 campaign pilot block (2026-09-25;
# 5% CAV penetration, 6 new worlds, 3 TRAIN + 3 TEST).  Identities 8401-8406:
# the first contiguous provably-unused run of the declared WM-41 candidate
# pool 8401-8600 (a fresh region strictly after the STAGE2 block 8001-8360;
# the 41xx/70xx/73xx/75xx families and every 92xxx fresh-world block are
# excluded by the audit's declared EXCLUDED_RANGES).  Exposure audit:
# output/campaign_pilot_v1/results_audit_wm41 (WM-41 package; 1759 used
# identities with provenance, all 200 pool members unused, chosen block
# 8401-8406).  Each seed is accepted ONLY with its own planned mainline
# loading from the registered WM-41 world plan (WORLD_PLAN.json, plan seed
# 20260928).  Pilot role only: consumed by the WM-41 runner preflight, never
# by a prospective (final-test) registration.
WM41_PILOT_SEEDS = frozenset(range(8401, 8407))
WM41_SEED_LOADING_MAP = {
    8401: 2700, 8402: 2700, 8403: 2700,
    8404: 2600, 8405: 2700, 8406: 2600,
}
# 2026-09-26: WM-43 wall-mechanics fixture block (PREPARED; consumed ONLY by the WM-43
# fixture preflight/runner).  5 identities 920244-920248: the first
# contiguous provably-unused run after the WM-34 fresh-world block
# (920144-920243) of the audited 92xxx series; the token 920244 occurs in
# the prior record ONLY as that block's exclusive end bound (declared
# non-use).  Exposure audit: output/action_model_v1/fixture/
# EXPOSURE_AUDIT_WM43.json (audit sha256 5d178eff3549efde967b77ac4302faf4064c314d1b1b8afee49edfabc9ae1a05).
# Each seed is accepted ONLY with its own registered fixture loading 2000
# (registry key only: the micro-roster scenario generates no stochastic
# demand).  Development-mechanics role; never a prospective (final-test) seed.
WM43_WALL_MECHANICS_SEEDS = frozenset(range(920244, 920249))
WM43_SEED_LOADING_MAP = {
    920244: 2000, 920245: 2000, 920246: 2000, 920247: 2000, 920248: 2000,
}
# 2026-09-30: WM-43B wall-closure re-validation fixture block (DRAFT; consumed ONLY by the WM-43B fixture
# preflight/runner, output/action_model_v1/fixture_v2).  25 identities
# 920249-920273: the block right after WM43_WALL_MECHANICS_SEEDS
# (920244-920248), 5 per arm A1..A5 in order.  Exposure audit:
# output/action_model_v1/fixture_v2/EXPOSURE_AUDIT_WM43B.json (INTEGRATOR
# must re-run exposure_audit_wm43b.py on the main tree before applying).
# Each seed is accepted ONLY with its own registered fixture loading 2000
# (registry key only: the micro-roster scenario generates no stochastic
# demand).  Development-mechanics role; never a prospective seed.
WM43B_WALL_CLOSURE_SEEDS = frozenset(range(920249, 920274))
WM43B_SEED_LOADING_MAP = {seed: 2000 for seed in range(920249, 920274)}


# WM-44 stage-1 chunk 1 (exposure-audited fresh; see seeds_py_wm44.diff)
WM44_CHUNK1_SEEDS = frozenset(range(8407, 8477))
WM44_CHUNK1_LOADING_MAP = {8407: 2400, 8408: 2600, 8409: 2600, 8410: 2700, 8411: 2700, 8412: 2700, 8413: 2600, 8414: 2000, 8415: 2700, 8416: 2000, 8417: 2700, 8418: 2600, 8419: 2600, 8420: 2000, 8421: 2600, 8422: 2700, 8423: 2400, 8424: 2400, 8425: 2400, 8426: 2600, 8427: 2600, 8428: 2700, 8429: 2700, 8430: 2700, 8431: 2700, 8432: 2700, 8433: 2700, 8434: 2600, 8435: 2700, 8436: 2700, 8437: 2600, 8438: 2600, 8439: 2700, 8440: 2600, 8441: 2700, 8442: 2400, 8443: 2600, 8444: 2600, 8445: 2600, 8446: 2700, 8447: 2700, 8448: 2700, 8449: 2400, 8450: 2700, 8451: 2700, 8452: 2700, 8453: 2600, 8454: 2600, 8455: 2700, 8456: 2700, 8457: 2700, 8458: 2700, 8459: 2700, 8460: 2700, 8461: 2700, 8462: 2600, 8463: 2700, 8464: 2600, 8465: 2700, 8466: 2700, 8467: 2600, 8468: 2600, 8469: 2400, 8470: 2000, 8471: 2700, 8472: 2700, 8473: 2700, 8474: 2600, 8475: 2000, 8476: 2600}


# WM-47 stage-1 chunk 2 (exposure-audited fresh; see seeds_py_wm47.diff)
WM47_CHUNK2_SEEDS = frozenset(range(8477, 8547))
WM47_CHUNK2_LOADING_MAP = {8477: 2600, 8478: 2700, 8479: 2700, 8480: 2600, 8481: 2700, 8482: 2000, 8483: 2700, 8484: 2600, 8485: 2700, 8486: 2700, 8487: 2400, 8488: 2400, 8489: 2700, 8490: 2700, 8491: 2700, 8492: 2600, 8493: 2600, 8494: 2600, 8495: 2700, 8496: 2700, 8497: 2400, 8498: 2600, 8499: 2700, 8500: 2700, 8501: 2600, 8502: 2700, 8503: 2700, 8504: 2600, 8505: 2700, 8506: 2600, 8507: 2000, 8508: 2700, 8509: 2700, 8510: 2700, 8511: 2000, 8512: 2700, 8513: 2600, 8514: 2600, 8515: 2700, 8516: 2600, 8517: 2700, 8518: 2400, 8519: 2600, 8520: 2700, 8521: 2700, 8522: 2000, 8523: 2600, 8524: 2600, 8525: 2700, 8526: 2700, 8527: 2700, 8528: 2400, 8529: 2600, 8530: 2600, 8531: 2700, 8532: 2600, 8533: 2600, 8534: 2600, 8535: 2400, 8536: 2400, 8537: 2600, 8538: 2600, 8539: 2700, 8540: 2700, 8541: 2700, 8542: 2700, 8543: 2700, 8544: 2700, 8545: 2000, 8546: 2700}


# WM-54 stage-1 chunk 3 (exposure-audited fresh; see seeds_py_wm54.diff)
WM54_CHUNK3_SEEDS = frozenset(range(8547, 8617))
WM54_CHUNK3_LOADING_MAP = {8547: 2700, 8548: 2600, 8549: 2600, 8550: 2600, 8551: 2000, 8552: 2700, 8553: 2700, 8554: 2700, 8555: 2700, 8556: 2400, 8557: 2700, 8558: 2700, 8559: 2700, 8560: 2700, 8561: 2700, 8562: 2700, 8563: 2600, 8564: 2600, 8565: 2600, 8566: 2700, 8567: 2700, 8568: 2600, 8569: 2400, 8570: 2700, 8571: 2700, 8572: 2600, 8573: 2000, 8574: 2700, 8575: 2700, 8576: 2700, 8577: 2600, 8578: 2600, 8579: 2700, 8580: 2700, 8581: 2600, 8582: 2600, 8583: 2700, 8584: 2700, 8585: 2700, 8586: 2700, 8587: 2700, 8588: 2700, 8589: 2600, 8590: 2400, 8591: 2600, 8592: 2600, 8593: 2700, 8594: 2600, 8595: 2400, 8596: 2400, 8597: 2600, 8598: 2700, 8599: 2600, 8600: 2000, 8601: 2700, 8602: 2700, 8603: 2600, 8604: 2600, 8605: 2600, 8606: 2000, 8607: 2600, 8608: 2400, 8609: 2600, 8610: 2000, 8611: 2400, 8612: 2700, 8613: 2700, 8614: 2700, 8615: 2700, 8616: 2700}


# WM-58 stage-1 chunk 4 (exposure-audited fresh; see seeds_py_wm58.diff)
WM58_CHUNK4_SEEDS = frozenset(range(8617, 8707))
WM58_CHUNK4_LOADING_MAP = {8617: 2700, 8618: 2000, 8619: 2700, 8620: 2700, 8621: 2700, 8622: 2700, 8623: 2600, 8624: 2600, 8625: 2700, 8626: 2700, 8627: 2700, 8628: 2700, 8629: 2000, 8630: 2700, 8631: 2600, 8632: 2000, 8633: 2000, 8634: 2400, 8635: 2700, 8636: 2700, 8637: 2600, 8638: 2700, 8639: 2700, 8640: 2000, 8641: 2700, 8642: 2600, 8643: 2600, 8644: 2700, 8645: 2400, 8646: 2600, 8647: 2000, 8648: 2600, 8649: 2600, 8650: 2700, 8651: 2400, 8652: 2700, 8653: 2600, 8654: 2600, 8655: 2400, 8656: 2600, 8657: 2700, 8658: 2600, 8659: 2700, 8660: 2700, 8661: 2600, 8662: 2600, 8663: 2700, 8664: 2600, 8665: 2700, 8666: 2700, 8667: 2700, 8668: 2600, 8669: 2700, 8670: 2700, 8671: 2700, 8672: 2700, 8673: 2700, 8674: 2600, 8675: 2700, 8676: 2700, 8677: 2700, 8678: 2700, 8679: 2400, 8680: 2600, 8681: 2700, 8682: 2400, 8683: 2400, 8684: 2700, 8685: 2700, 8686: 2600, 8687: 2600, 8688: 2700, 8689: 2600, 8690: 2600, 8691: 2700, 8692: 2600, 8693: 2600, 8694: 2600, 8695: 2400, 8696: 2600, 8697: 2700, 8698: 2600, 8699: 2700, 8700: 2600, 8701: 2700, 8702: 2700, 8703: 2600, 8704: 2700, 8705: 2700, 8706: 2400}


# WM-58 stage-1 chunk 5 (WM-N1 sealed TEST set; see seeds_py_wm_c5.diff)
WM58_CHUNK5_SEEDS = frozenset(range(8707, 8737))
WM58_CHUNK5_LOADING_MAP = {8707: 2400, 8708: 2400, 8709: 2400, 8710: 2400, 8711: 2400, 8712: 2400, 8713: 2400, 8714: 2400, 8715: 2400, 8716: 2400, 8717: 2600, 8718: 2600, 8719: 2600, 8720: 2600, 8721: 2600, 8722: 2600, 8723: 2600, 8724: 2600, 8725: 2600, 8726: 2600, 8727: 2700, 8728: 2700, 8729: 2700, 8730: 2700, 8731: 2700, 8732: 2700, 8733: 2700, 8734: 2700, 8735: 2700, 8736: 2700}


# WM-N1-REPL Part R fresh TEST_REPL block (
# registration runs/active/WM-N1-REPL-CL-20261005/REGISTRATION.md, seed audit
# SEED_AUDIT.json).  30 identities 930001-930030: exact-token audit found no
# prior use (highest reserved identity 920273 + the PLWM1 98xx block).  Each
# seed is accepted ONLY with its own planned loading from
# output/campaign_stage1_v3/results_plan_repl/WORLD_PLAN.json (plan seed
# 20261013).  Sealed TEST_REPL role; outcomes read only by the registered
# scoring pass.
WM_N1_REPL_SEEDS = frozenset(range(930001, 930031))
WM_N1_REPL_SEED_LOADING_MAP = {
    930001: 2600, 930002: 2600, 930003: 2400, 930004: 2600,
    930005: 2400, 930006: 2700, 930007: 2700, 930008: 2700,
    930009: 2000, 930010: 2600, 930011: 2600, 930012: 2600,
    930013: 2700, 930014: 2700, 930015: 2600, 930016: 2700,
    930017: 2600, 930018: 2600, 930019: 2700, 930020: 2700,
    930021: 2000, 930022: 2700, 930023: 2700, 930024: 2700,
    930025: 2400, 930026: 2700, 930027: 2600, 930028: 2700,
    930029: 2700, 930030: 2700,
}
# PL-WM1 fresh block (planner; exposure audit PLWM1_SEED_AUDIT.json, worlds PLWM1_WORLDS.json 3a88485c; see seeds_py_plwm1.diff)
PLWM1_SEEDS = frozenset(range(9801, 9821))
PLWM1_LOADING_MAP = {9801: 2700, 9802: 2600, 9803: 2700, 9804: 2700, 9805: 2700, 9806: 2600, 9807: 2600, 9808: 2400, 9809: 2700, 9810: 2700, 9811: 2700, 9812: 2600, 9813: 2600, 9814: 2400, 9815: 2700, 9816: 2700, 9817: 2700, 9818: 2600, 9819: 2600, 9820: 2400}
# C4 GENRAND undirected random-action WM TRAINING data block (2026-10-14, draft registration
# GENRAND_REGISTRATION.md; the integrator registers the range and runs the holdout-guard
# membership check before collection).  TRAINING only: never evaluation, CAL or HOLDOUT.
C4_GENRAND_SEEDS = frozenset(range(961000, 962000))
C4_GENRAND_LOADINGS = (2400, 2700)
# C4 WALLREGEN unseen-wall EVALUATION block (2026-10-14, WALLREGEN_NOTES.md): fresh worlds for the
# regenerated unseen-wall world-model evaluation sets (new driver population).  EVALUATION only:
# never WM/policy training, never CAL/HOLDOUT/TEST.  Proposed by the wallregen worker after a scan of
# the seed tables, protocols/experiment_history.json and runs/active/**/eval_seeds.json (no hit); the
# integrator registers the block and runs the holdout-guard membership check before collection.
C4_WALLREGEN_EVAL_SEEDS = frozenset(range(962000, 962200))
# loadings: any multiple of 50 veh/h in [1000, 2800] (demand levels are an design choice; the
# textbook population saturates near 1930 veh/h)
C4_WALLREGEN_EVAL_LOADINGS = tuple(range(1000, 2801, 50))
# C4 V5 blocks (2026-10-14): textbook-population
# v5 WM/policy TRAINING data (native + generic_random + paired branches + MAPPO fine-tune) and the
# final v5 policy EVALUATION worlds.  Scan of the seed tables and the ledger: no overlap.
C4_V5_TRAIN_SEEDS = frozenset(range(963000, 964000))
C4_V5_EVAL_SEEDS = frozenset(range(964000, 964200))
C4_V5_LOADINGS = tuple(range(1900, 2801, 50))
# C4 V6 long-branch TRAINING block: 120-300 s paired SUMO branches
# (undirected random arms + native) for the v6 WM. TRAINING only. Scan of seed tables + ledger: no overlap.
C4_V6_LONGPAIR_SEEDS = frozenset(range(965000, 966000))
# C4 Dreamer-loop TRAINING block: SUMO episodes collected with the learned
# policy for continuous WM refresh. TRAINING only. Scan of seed tables + ledger: no overlap.
C4_LOOP_TRAIN_SEEDS = frozenset(range(966000, 970000))
# Every partition that may not enter a future prospective (final-test)
# evaluation. FINAL_TEST_SEEDS is deliberately absent: it is the only
# partition reserved for prospective evaluation.
PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS = tuple(
    sorted(
        set(DEVELOPMENT_SEEDS)
        | set(CONTROLLER_FEATURE_VALIDATION_SEEDS)
        | set(OE09_DEVELOPMENT_SEEDS)
        | set(VALIDATION_SEEDS)
        | set(WALL_DEVELOPMENT_SEEDS)
        | set(PREDICTOR_DEVELOPMENT_SEEDS)
        | set(PREDICTOR_VALIDATION_SEEDS)
        | set(SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS)
        | set(SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_SEEDS)
        | set(SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_SEEDS)
        | set(STAGE1B_GENERATION_SEEDS)
        | set(STAGE1C_RELEASE_SEEDS)
        | set(STAGE2_GENERATION_SEEDS)
        | set(CV20_MULTISEED_CORPUS_SEEDS)
        | set(CV28_VALIDATION_HOLDOUT_SEEDS)
        | set(CV29_CORPUS_EXTENSION_SEEDS)
        | set(CV31_COVERAGE_EXTENSION_SEEDS)
        | set(NATIVE_DAMPING_TRAINING_SEEDS)
        | set(NATIVE_DAMPING_DEVELOPMENT_SEEDS)
        | set(NATIVE_DAMPING_ACCEPTANCE_SEEDS)
        | set(OE17_DURATION_SELECTION_SEEDS)
        | set(OE32_OBSERVATION_DEVELOPMENT_SEEDS)
        | set(OE59_CONDITIONAL_PANEL_SEEDS)
        | set(OE62_CONDITIONAL_SCREEN_SEEDS)
        | set(OE65_DATASET_SEEDS)
    )
)

SEALED_SEEDS = frozenset(VALIDATION_SEEDS + PREDICTOR_VALIDATION_SEEDS + FINAL_TEST_SEEDS
                         + NATIVE_DAMPING_ACCEPTANCE_SEEDS)
EXPOSED_CONTROLLER_SEEDS = (frozenset(range(4107, 4133)) | frozenset(OE09_DEVELOPMENT_SEEDS)
                            | frozenset(OE17_DURATION_SELECTION_SEEDS))
TRAINING_LOADS = (1900,2000,2100,2600,2200,1900,2000,2700,2100,2200,1900,2600,2000,2100,2200,2700)
DEVELOPMENT_LOADS = (2000,2000,2200,2200,2600,2600,2700,2700)


def require_development_case(seed, loading, role):
    """Only previously exposed direct-fixture cases; no new campaign authorization."""
    if type(seed) is not int or type(loading) is not int:
        raise ValueError("seed and loading must be exact integers, not bools or float aliases")
    if seed in SEALED_SEEDS:
        raise ValueError("reserved acceptance seed remains sealed")
    if role == "development" and seed in OE32_OBSERVATION_DEVELOPMENT_SEEDS and loading == 2700:
        return
    if role == "development" and seed in OE34_RELATIONAL_DEVELOPMENT_SEEDS and loading == 2700:
        return
    if role == "development" and seed in OE59_CONDITIONAL_PANEL_SEEDS and loading == 2700:
        return
    if role == "development" and seed in OE62_CONDITIONAL_SCREEN_SEEDS and loading == 2700:
        return
    if role == "development" and seed in OE65_DATASET_SEEDS and loading in (1900, 2000, 2100, 2200, 2600, 2700):
        return
    if role == "exposed" and seed in EXPOSED_CONTROLLER_SEEDS:
        return
    if role == "exposed" and seed in CV20_MULTISEED_CORPUS_SEEDS and loading in (2000, 2700):
        return
    if role == "exposed" and seed in CV28_VALIDATION_HOLDOUT_SEEDS and loading in (2000, 2700):
        return
    if role == "exposed" and seed in CV29_CORPUS_EXTENSION_SEEDS and loading in (2000, 2700):
        return
    if role == "exposed" and seed in CV31_COVERAGE_EXTENSION_SEEDS and loading in (2400, 2700):
        return
    if role == "exposed" and seed in WM28_FRESH_VALIDATION_SEEDS \
            and loading == WM28_SEED_LOADING_MAP.get(seed):
        return
    if role == "exposed" and seed in WM34_FRESH_RETEST_SEEDS \
            and loading == WM34_SEED_LOADING_MAP.get(seed):
        return
    if role == "exposed" and seed in WM41_PILOT_SEEDS \
            and loading == WM41_SEED_LOADING_MAP.get(seed):
        return
    if role == "exposed" and seed in WM43_WALL_MECHANICS_SEEDS \
            and loading == WM43_SEED_LOADING_MAP.get(seed):
        return
    if role == "exposed" and seed in WM43B_WALL_CLOSURE_SEEDS \
            and loading == WM43B_SEED_LOADING_MAP.get(seed):
        return

    if role == "exposed" and seed in C4_GENRAND_SEEDS and loading in C4_GENRAND_LOADINGS:
        return

    if role == "exposed" and seed in C4_WALLREGEN_EVAL_SEEDS and loading in C4_WALLREGEN_EVAL_LOADINGS:
        return

    if role == "exposed" and seed in (C4_V5_TRAIN_SEEDS | C4_V5_EVAL_SEEDS) and loading in C4_V5_LOADINGS:
        return

    if role == "exposed" and seed in C4_V6_LONGPAIR_SEEDS and loading in C4_V5_LOADINGS:
        return

    if role == "exposed" and seed in C4_LOOP_TRAIN_SEEDS and loading in C4_V5_LOADINGS:
        return

    if role == "exposed" and seed in WM44_CHUNK1_SEEDS \
            and loading == WM44_CHUNK1_LOADING_MAP.get(seed):
        return

    if role == "exposed" and seed in WM47_CHUNK2_SEEDS \
            and loading == WM47_CHUNK2_LOADING_MAP.get(seed):
        return

    if role == "exposed" and seed in WM54_CHUNK3_SEEDS \
            and loading == WM54_CHUNK3_LOADING_MAP.get(seed):
        return

    if role == "exposed" and seed in WM58_CHUNK4_SEEDS \
            and loading == WM58_CHUNK4_LOADING_MAP.get(seed):
        return

    if role == "exposed" and seed in WM58_CHUNK5_SEEDS \
            and loading == WM58_CHUNK5_LOADING_MAP.get(seed):
        return

    if role == "exposed" and seed in WM_N1_REPL_SEEDS \
            and loading == WM_N1_REPL_SEED_LOADING_MAP.get(seed):
        return

    if role == "exposed" and seed in PLWM1_SEEDS \
            and loading == PLWM1_LOADING_MAP.get(seed):
        return
    allowed = {"training": tuple(zip(NATIVE_DAMPING_TRAINING_SEEDS, TRAINING_LOADS)),
               "development": tuple(zip(NATIVE_DAMPING_DEVELOPMENT_SEEDS, DEVELOPMENT_LOADS))}
    if (seed, loading) not in allowed.get(role, ()):
        raise ValueError("unregistered or unexposed development case")


def exclusion_provenance():
    seeds = list(PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS)
    return dict(source="archive/pre_circles/code/idm/contract.py",
                excluded_seeds=seeds, sealed_seeds=sorted(SEALED_SEEDS),
                exclusions_sha256=hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
                absence_proves_freshness=False)


def verify_prospective_registration(path):
    """Read an explicit audited registration; this never generates scenarios."""
    registration = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = registration["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("prospective seed identities must be unique integers")
    if set(seeds) & (set(PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS) | SEALED_SEEDS):
        raise ValueError("prospective registration reuses excluded/reserved seeds")
    if not registration.get("exposure_audit_sha256") or registration.get("audit_passed") is not True:
        raise ValueError("prospective registration requires exposure-audit provenance")
    return registration
