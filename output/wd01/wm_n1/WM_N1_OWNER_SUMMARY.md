# WM-N1 Idea A in plain language

## What it is

One "what happens next" model of the road, built the way we actually drive
the fleet. To look one step ahead (half a second) it does four things:

1. Every CAV is moved by our REAL controller code (native, damping, PI, or
   the wall plan). The controllers are never replaced by the network.
2. Every ordinary car is moved by the current rule-based driver model PLUS a
   small learned correction. That correction only sees where cars are, how
   fast they go, and which lane they are in — never a strategy name, never
   an action label, never a "this is a CAV" flag.
3. New cars arrive through a learned arrival model, driven by the ramp
   counters.
4. A safety layer keeps the result physical: no negative speeds, no
   overlapping cars, no illegal lane changes.

Repeating that step plays whole futures forward, so we can ask the planner's
question: "if I choose plan P, what happens over the next minutes, and how
much better is it than doing nothing?"

## How it answers the planner's question

The long-horizon answer (delay, risky moments, hard braking) comes from the
rollout plus a small correction head. That head is fed only a summary of the
CAVs' PLANNED movement — where our controllers intend to drive — never a
label like "this was the wall arm". Two different plans that produce the
same car movement look identical to it. It also produces an "unfamiliar
motion" score, calibrated on tuning data only, so the planner can hold back
when the fleet starts moving in ways the model never saw.

## Why we believe it will generalise

Because CAVs are moved by the real controllers, a plan the model has never
seen in training is still executed by code it can trust. The network only
has to learn how ordinary drivers respond. We test exactly that with
held-out-strategy trials: train without any wall law, test on the new wall
law; train without PI, test on PI; train without one wall speed, test on it.

## Keeping it honest

- It never sees hidden driver types, the true CAV share, or any outcome it
  is not allowed to know.
- Training data, tuning data (CAL) and the final test are separate. The
  final test is a brand-new chunk 5 of simulated worlds, sealed until one
  single reading.
- Five network members are trained with different seeds; we report how sure
  they are collectively. Member 1 is trained first so we get an early
  working model, but the final test refuses to run until all five exist.
- Known wrinkles, disclosed in the design: the wall training data ran under
  the older wall law (with a relabelled road-end event), short planned
  actions are rare in the data until the J-PULSE runs land, and the
  controller-parity pre-flight check waits on one missing code piece.

## The four versions we report

- A: the new controller-in-the-loop model (the headline).
- N: the older pure-network design (kept as a comparison).
- T: the current tree model.
- C: a blend of the network and tree answers.

The planner picks among them on the tuning set BEFORE the test. If two are
equally good we keep the simpler one, in this order: tree, A, blend, N.

## How we decide it worked

On the sealed test set, the chosen version must predict the DIFFERENCE
between strategies better than today's model, with statistical confidence,
while not being worse at the basics (overall delay, short-horizon positions,
crashes). Otherwise today's model stays.

## What it needs

- 3–4 rented GPU boxes (final count after a 1-hour speed test), one job per
  box; about 70–130 GPU-hours all in.
- The training data is already featurised from past runs; no new simulation
  is needed for training.
- Six pre-flight checks run first on real training data (replay fidelity,
  controller parity, leak checks, tiny-scale training, an end-to-end smoke,
  and dry-running the held-out splits). Five pass today; the controller
  parity check waits on its code piece.

## Paper framing

The model is trained and evaluated on the TRUE simulator state. Rebuilding
its inputs from sparse roadside sensing is future work: the design keeps an
interface for it, but nothing of it is built or claimed.

(As-built details and citations: WM_N1_IDEA_A_DESIGN.md, same directory.)
