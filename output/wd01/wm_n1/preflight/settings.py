"""Gate settings (paths + CPU/device policy)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WM_N1_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIRNAME = "reports"


@dataclass
class Settings:
    """Shared gate settings.  CPU only; no SUMO; TRAIN worlds only."""

    wm_n1_dir: Path = WM_N1_DIR
    reports_dir: Path = WM_N1_DIR / "preflight" / DEFAULT_REPORTS_DIRNAME
    device: str = "cpu"
    dry: bool = False

    def resolve_report_dir(self) -> Path:
        return Path(self.reports_dir)
