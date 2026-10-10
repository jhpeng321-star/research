"""Traffic Shepherd: frozen CIRCLES damping and CAV-reported escalation interfaces."""
from pathlib import Path

__version__ = "1.0.0"

# Superseded modules were relocated unchanged (MIG-03). Keeping them importable under
# their original names lets frozen historical scripts replay; active modules must not
# import them.
_ARCHIVED = Path(__file__).resolve().parents[1] / "archive" / "circles_era" / "traffic_shepherd"
if _ARCHIVED.is_dir():
    __path__.append(str(_ARCHIVED))
