export WM_N1_PAIRED_EFFECT_CLIMATOLOGY=E:/research-dense/runs/active/WM-N1-SCORE-A9-20261005/TRAIN_CLIMATOLOGY.json
cd E:/rd-wt/rollout-entry
exec /c/Users/jpeng/miniconda3/envs/rl_driving/python.exe output/wd01/wm_n1/closedloop/runner.py --world w8676m2700p20t360 --policy pl01_a11 --plan E:/research-dense/output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json --bases-root E:/research-dense/runs/campaign_stage1_chunk4 --out E:/rd-wt/rollout-entry/_out/smoke_open_w8676m2700p20t360_pl01_a11 --horizon-s 180 --threads 8 --code-root E:/research-dense --open-road
