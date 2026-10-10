import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, r"E:/rd-wt/wd01v32f/output/wd01/partB_v0data")
    sys.path.insert(0, r"E:/rd-wt/wd01v32f/output/wd01")
    import build_v032
    sys.exit(build_v032.main([
        "--data-root", "E:/research-dense",
        "--workers", "4",
    ]))
