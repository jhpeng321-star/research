# IDM v5.6 Candidate B support-CAV availability audit

Status: **stopped before outcome execution: treatment-blind support cohort not established**

The current overlay reserves every nonmember CAV inside its 200--1000 m
recruitment band.  A valid Candidate-B support cannot be a wall member or
reserve, so it must be outside that band.  The premerge edge is only 1144.95 m
long, leaving at most 145 m upstream of the recruitment band.  At t=880, the
four cases have 42--56 active corridor CAVs and feasible single walls, but
the saved artifacts do not retain the outside-band positions, gaps, TTC,
runway, or 20 s observability needed to select a safe per-lane support.

No existing implementation has a damping-preserving support role: nonmembers
remain exact damping, while the historical pre-band `WallHoldPolicy` path does
not preserve that contract.  Selecting a support band now would invent a
treatment definition without non-outcome evidence.  Candidate B is therefore
mechanically unqualified under this frozen contract; no outcome branches were
launched.
