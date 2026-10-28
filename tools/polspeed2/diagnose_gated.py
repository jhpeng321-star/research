from pathlib import Path

root = Path('/root/autodl-tmp/c4/polspeed2_20261028/qualification3_tmp/test_gated_vjps_preserve_full_1/future_workers')
for path in sorted(root.glob('*/worker*.log')):
    print(path, path.read_text(), flush=True)
for path in sorted(root.glob('*/*/*.error')):
    print(path, path.read_text(), flush=True)
