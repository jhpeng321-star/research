# ACTION_REPRESENTATION

Disclosure: the TRAIN planned-schedule comparison set is dominated by whole-window actions (4758/4952 = 96.1% whole-window); short commitments are 5 pair(s) at <=60 s and 189 at 60-240 s, and 2 pair(s) differ solely in the first 60 s.  The per-CAV switching schedules and the J4 wall arms are the only source of start/stop variation; registration and report must state this.

Schema `wm_n1_action_representation/1`; generated 2026-09-30T19:27:49Z.

Method: canonical planned-schedule descriptors (19 x 30 s per CAV/group + wall hold/speed/cap); all unordered distinct-descriptor sibling pairs per TRAIN world; duration = onset-to-last differing span.

## TRAIN schedule-pair strata (aggregate)

### Duration

| stratum | pairs |
| --- | --- |
| <=60s | 5 |
| 60-240s | 189 |
| whole-window | 4758 |
| first-60-only | 2 |

### Onset

| bin | pairs |
| --- | --- |
| 0-30s | 4913 |
| 30-60s | 0 |
| 60-240s | 39 |
| after-240s | 0 |

### Per source

| source | TRAIN worlds | runs | distinct descriptors | pairs | <=60s | 60-240s | whole-window | first-60-only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chunk1 | 49 | 343 | 343 | 1029 | 0 | 0 | 1029 | 0 |
| chunk1_fix27 | 10 | 70 | 70 | 210 | 0 | 0 | 210 | 0 |
| chunk2 | 49 | 343 | 343 | 1029 | 0 | 0 | 1029 | 0 |
| chunk3 | 49 | 343 | 343 | 1029 | 0 | 0 | 1029 | 0 |
| chunk4 | 63 | 441 | 441 | 1323 | 0 | 0 | 1323 | 0 |
| j4_walls | 39 | 156 | 156 | 234 | 5 | 189 | 40 | 2 |
| truearms_chunk2 | 49 | 98 | 98 | 49 | 0 | 0 | 49 | 0 |
| truearms_chunk3 | 49 | 98 | 98 | 49 | 0 | 0 | 49 | 0 |

### Duration x onset (aggregate)

| stratum | pairs |
| --- | --- |
| 60-240s|0-30s | 153 |
| 60-240s|60-240s | 36 |
| <=60s|0-30s | 2 |
| <=60s|60-240s | 3 |
| whole-window|0-30s | 4758 |

