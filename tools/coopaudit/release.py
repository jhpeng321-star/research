"""Fail-closed, hash-bound release of the owner's artifact-check HOLD.

This module never clears a HOLD or starts a process. A reviewed release must be
registered separately after the upstream artifact result and corrected design.
"""
import hashlib
import json
from pathlib import Path
import re

from tools.coopaudit.design import CONTROLLERS


def require_release(root, selection=None, artifact=None):
    root = Path(root)
    if (root/'ARTIFACT_HOLD.json').exists():
        hold = json.loads((root/'ARTIFACT_HOLD.json').read_text())
        if hold.get('active') is not False:
            raise ValueError('owner artifact HOLD active; explicit reviewed release required')
    if (root/'YIELD').exists() or (root/'STOP').exists():
        raise ValueError('owned YIELD/STOP prevents dispatch')
    path = root/'ARTIFACT_RELEASE.json'
    if not path.exists():
        raise ValueError('missing reviewed artifact release; no audit SUMO authorised')
    release = json.loads(path.read_text())
    if release.get('status') != 'PASS' or release.get('registered') is not True:
        raise ValueError('artifact release must be PASS and registered')
    if release.get('controllers') != list(CONTROLLERS):
        raise ValueError('release does not match amended primary controller scope')
    if release.get('reserved_controller_count', 0) < len(CONTROLLERS)+1:
        raise ValueError('paired trained-policy budget reservation required')
    doc = Path(artifact) if artifact is not None else root/'ARTIFACT_CHECK.md'
    if not doc.is_file():
        raise ValueError('upstream artifact-check evidence missing')
    status_lines = re.findall(r'^\s*Status:\s*(PASS|FAIL|HOLD)\s*$',
                              doc.read_text(encoding='utf-8'), re.MULTILINE)
    status_lines += re.findall(r'^\s*#{1,6}\s+.*artifact check\s*[—–:-]\s*(PASS|FAIL|HOLD)\s*$',
                               doc.read_text(encoding='utf-8'), re.MULTILINE | re.IGNORECASE)
    if not status_lines or any(s.upper() != 'PASS' for s in status_lines):
        raise ValueError('artifact evidence requires an explicit PASS status without HOLD/FAIL')
    if hashlib.sha256(doc.read_bytes()).hexdigest() != release.get('artifact_document_sha256'):
        raise ValueError('artifact-check evidence changed; re-review required')
    if selection is None:
        selection = json.loads((root/'SELECTED_DESIGN.json').read_text())
    digest = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    if digest != release.get('selected_design_sha256'):
        raise ValueError('selected design changed after artifact release')
    if not release.get('source_commit') or not release.get('registration_entry'):
        raise ValueError('source and registration provenance required')
    return release
