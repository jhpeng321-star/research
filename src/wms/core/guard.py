"""Detection of restricted (sealed / calibration / holdout) path tokens.

The data layout forbids ever opening such material; every resolver and config
validator calls :func:`find_restricted_token` before a path is used.
"""
from __future__ import annotations

import re
from typing import Optional

from .errors import TrainDataRefusal

# Token boundaries are anything but [a-z0-9], so "cal" matches a path part named
# "cal" or "x-cal-y" but not "local" or "scale".
_RESTRICTED = re.compile(
    r"(?:^|[^a-z0-9])(cal(?:ibration)?|cal_split_a4|holdout|sealed|chunk[-_]?5|j4b|inc1)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


def find_restricted_token(text: str) -> Optional[str]:
    """Return the first restricted token found in ``text`` (case-insensitive), else ``None``."""
    match = _RESTRICTED.search(str(text).replace("\\", "/"))
    return match.group(1).lower() if match else None


def refuse_restricted(text: str, *, what: str = "path") -> None:
    """Raise :class:`TrainDataRefusal` when ``text`` contains a restricted token."""
    token = find_restricted_token(text)
    if token is not None:
        raise TrainDataRefusal(f"{what} {text!r} contains restricted token {token!r}; refusing")
