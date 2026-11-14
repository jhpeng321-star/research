# IDM v5.7 extended-control-zone source audit

Status: **treatment-blind geometry audit complete; one derived geometry frozen.**

> Correction: this v5.7 derived-network design was **not executed**. The
> added 900 m edge is outside its 200--1,900 m eligibility band and therefore
> changes upstream storage/arrival timing rather than active wall-control
> distance. It is retained as provenance only; v5.7.1 tests the band/duration
> hypothesis on the original corridor without this confound.

## What is and is not being changed

The original two-lane upstream edge `1191554636` is 2,644.22 m long in the source SUMO network. The merge edge, ramp, downstream bottleneck, lane counts, speed limits, IDM parameters, demand process, penetration and seeds are not altered. The existing v5.6.2 negative result remains intact.

The relevant limitation was narrower than the raw network: the prior compact wall only admitted members in the 200--1,000 m distance-to-merge band. With a 50 m release boundary, this supplies at most 800 m of eligible coordinated travel, irrespective of the longer raw edge.

## Treatment-blind kinematics

The v5.6.2 controller audit recorded activation member speeds of 24.57--27.63 m/s (upper observed value: 27.63 m/s). It used one second of assembly, approximately 14 s of HOLD and five seconds of taper. At a representative 25.5 m/s this is about 26 m + 357 m + 128 m = 511 m. That short pulse fits inside the old 800 m active band, but it cannot test gradual low-authority formation plus a 40 s HOLD.

For v5.7, the frozen requested lifecycle is 10 s assembly + 40 s HOLD + 8 s taper = 58 s. A conservative treatment-blind upper bound uses the fastest observed member without assuming that safety projection delivers any braking: 27.63 m/s x 58 s = 1,602.5 m. Adding the existing 200 m merge buffer yields 1,802.5 m. The one frozen usable boundary is therefore rounded upward to 1,900 m, yielding a 1,700 m eligible band (200--1,900 m). This rounding was done before v5.7 wall outcomes and is not a length sweep.

The derived scenario adds exactly 900 m of two-lane straight approach upstream of the original source node. It makes the available coordination approach longer (3,544.22 m total) while the expanded 1,900 m eligibility boundary is the actual mechanism under test. This audit does **not** claim that raw road length alone caused the v5.6.2 no-go; it tests whether its short eligible window/duration was a material limitation.

## Interpretation guardrail

The extension delays arrival at the unchanged merge. Consequently every v5.7 comparison is event-aligned to the treatment-blind first safe full-lane distance-band opportunity, has an identical 178 s terminal horizon within its pair, and separately reports bottleneck arrivals and completed service. Extra upstream storage or delayed merge arrival is never a wall benefit.
