# WM-N1 featurizer micro-smoke (chunk-4 TRAIN, ≤2 runs, 1 process)

Zero SUMO.  One process, BelowNormal priority, 1 worker.

Command (cwd `output/wd01/wm_n1`):

```
python -m features featurize --chunk chunk4 --worlds w8623m2600p02t540 \
    --arms base,native --data-root output/wd01/wm_n1/features/data --workers 1
```

stdout:

```
featurize: runs=2 written=2 skipped=0 wall=48.8s
  [written] chunk4/w8623m2600p02t540__base n_ticks=2281 bytes=14741859 (14.742 MB) pair_key=w8623m2600p02t540@t0=540
  [written] chunk4/w8623m2600p02t540__native n_ticks=1201 bytes=8147250 (8.147 MB) pair_key=w8623m2600p02t540@t0=540
```

verify:

```
python -m features verify --data-root output/wd01/wm_n1/features/data
{ "ok": true, "n_runs": 2, "errors": [] }
```

The two shards are the base and the sibling `native` arm of the same world
(`w8623m2600p02t540`, t0=540) and share the `pairs` entry
`w8623m2600p02t540@t0=540` in `INDEX.json`.
