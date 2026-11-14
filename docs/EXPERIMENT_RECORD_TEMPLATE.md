# Experiment record template (required for registering and closing experiments)

Every new experiment must produce (1) a novelty check, (2) a registration before
execution, and (3) a closing record appended to
[protocols/experiment_history.json](../protocols/experiment_history.json) (schema
`experiment_history/2`) and, when it answers a question, to
[EXPERIMENT_HISTORY](EXPERIMENT_HISTORY.md) and [OPEN_QUESTIONS](OPEN_QUESTIONS.md).
Keep entries compact; link evidence, do not copy artifacts. Never read sealed raw
outcomes; record them by permitted metadata only.

## Step 0 — pre-run novelty check (required before proposing any simulation)

```powershell
python tools/experiment_history.py closest --text "<your question in plain words>"
python tools/experiment_history.py search --text "<axis, e.g. 'target speed HOLD'>"
python tools/experiment_history.py card <closest prior ID>
python tools/experiment_history.py open-questions
python tools/experiment_history.py pending
```

The registration must name the closest prior experiment and state the **exact
unanswered difference** (mechanism addressed, seed groups, observation contract,
action space, or decision frame — entry vs continuation). If no such difference
exists, the run would repeat an answered question: stop.

## Registration (before any simulation)

```markdown
### REGISTRY ENTRY — <STABLE-ID>
- Date registered / family / name:
- Scientific question / hypothesis:
- Status: proposed | registered (not yet executed)
- Record kind: experiment | analysis | mechanics | protocol | campaign
- Controller/actuator + source/version identity (hashes, policy version):
- Damping baseline and car-following model:
- Network/scenario; exposed seed/split metadata (permitted fields only):
- Runtime observation contract; any privileged information (must be none online):
- Decision frame: entry_vs_policy | continuation_keep_vs_adjust | continuation_keep_vs_release
- Changed variables; fixed variables; known confounders:
- Comparators; exact matching/reuse evidence (which prior runs are reused and how verified):
- Requested assembly/HOLD/exit and target speeds (nominal, before delivery):
- Registered budget (registered ceiling; allocation/charging rules), stopping rules, retry policy:
- Original criteria (frozen, quoted verbatim; never weakened later):
- Closest prior experiment(s) and the exact unanswered difference (from Step 0):
- Immutable registration references (commit SHA of the ledger update, registration file
  SHA-256, protocol path frozen before outcomes):
- Paths: protocol/registration file, code, planned output directory:
```

## Closing record (after completion, failure, or stop)

```markdown
### CLOSE-OUT — <STABLE-ID>
- Final status: attempted | completed | invalid | stopped | superseded
  (execution status, evidence validity and scientific verdict recorded as SEPARATE
  structured fields in the ledger entry)
- Requested vs DELIVERED assembly/HOLD/exit and target speeds (actuals):
- Actual attempts and charged simulated seconds (incl. preserved failures);
  reconcile allocation-by-allocation against the registered budget; classify every
  allocation as completed | failed-charged | reused (0 s) | trace-equivalent (0 s)
  | zero-simulation re-verification:
- Physical results: structured outcomes with cost difference = treatment cost -
  reference cost (NEGATIVE = improvement), unit and normalization denominator per
  metric; if an "improvement" value is also stored it must equal the negative of the
  cost difference (the validator checks this). Where feasible, extract the numbers
  programmatically from the registered analysis JSON rather than retyping them, and
  set verified_against to that file path.
- Original criteria and ORIGINAL verdict (do not reinterpret):
- Counting: executed treatments vs delivered walls vs refusals; rows/branches vs
  independent seed groups; exact equivalence vs small aggregate differences:
- Integrity limitations, uncertainty, generalization limits:
- Answers the question: fully | partially | not at all:
- Scoring/verifier failures and their resolution receipts; distinguish
  re-verification (0 new simulation) from new physical runs; a STOP/crash becomes a
  preserved charged failure or a separately registered repair - never a silent retry:
- Relationships: campaign / allocated run / completed trajectory / comparison /
  derived analysis / reused reference, recorded in the ledger's runs and comparisons
  registries (a run appearing in several comparisons stays ONE sample and ONE charge):
- What must NOT be inferred from this result:
- Paths: analysis, summary, receipts, report:
- Ledger updated (protocols/experiment_history.json entry + registries): yes
- OPEN_QUESTIONS updated (which OQ entries / uncertainty types changed): 
- ROUTE.md updated: yes/no   EXPERIMENT_HISTORY.md updated: yes/no
```

## Current-status summary (keep at the top of the ledger `coverage` block)

One line per close-out: date, ID, one-sentence verdict, and whether any open question
changed state. The ledger validation (`python tools/experiment_history.py validate`,
exit code 0 required) and `tests/test_experiment_history_ledger.py` must pass after
every close-out; they catch duplicate IDs, broken reuse/relationship links, missing
paths, sign-arithmetic violations, reused trajectories counted as independent
evidence, overlapping budget totals, and numeric summaries without a cited
authoritative artifact.

## Rules

The integrating agent owns close-out, including edits contributed by parallel
agents. Follow AGENTS.md's required close-out for all scientific analyses as well
as simulations. Save routine test commands/results with the associated task.
Before final delivery, report the stable experiment/analysis IDs, artifact paths,
ledger update and unresolved verification items. For interruption or failure,
record status and consumed allocation promptly; do not wait for a successful run.

Close-out commands (existing rl_driving environment):

```powershell
python tools/experiment_history.py validate
python tools/experiment_history.py verify-sources
python tools/experiment_history.py summary --write
python tools/experiment_history.py summary --check
python -m pytest -q tests/test_experiment_history_ledger.py
```

Source verification is limited to registered permitted development artifacts.
Unknown budgets/review gaps remain explicit; no frozen evidence is modified to
make a check pass. These checks validate records and mapped sources, not automatic
scientific acceptance or complete coverage of all files on disk.

1. Consult the ledger before proposing simulations; run the Step 0 novelty check and
   cite the closest prior experiment.
2. Register source, splits, comparators and budget before execution; freeze the
   registration (hash or commit reference) before outcomes exist; no retrospective
   registrations for new physical runs.
3. Append outcomes, failures, reuse and stop decisions after completion — invalid
   and stopped runs are recorded, never deleted.
4. Distinguish executed treatment from proposed; nominal duration from delivered
   HOLD; feasibility from incremental benefit; invalid from valid negative; refused
   wall from executed wall; rows/branches from independent seed groups; exact
   equivalence from small aggregate differences; current moving-wall evidence from
   older fixed-band metering; development findings from held-out evidence; measured
   effects from temporal localization, plausible mechanisms and experimentally
   isolated causes.
5. When prose later disagrees with machine-readable evidence: preserve the original
   text, add a correction entry naming the conflicting claim, cite the supporting
   artifact, and state whether the verdict is affected.
6. This is recordkeeping, not an approval bureaucracy: ordinary read-only work needs
   no registration — only new simulations, training, or acceptance-adjacent access.
