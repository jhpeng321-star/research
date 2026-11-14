"""WD-01-PA2G-LC2 structural lane-change variants (OFF by default).

V-a persistence, V-b manoeuvre time, V-c strategic early merge and
V-abc, exactly as registered in REGISTRATION_PA2G_LC2.md section 2.
The registered a-priori values live in variants_lc2.json (NOT in
params_nominal.json, which stays byte-identical).

Mechanism: `variant_params(params, variant)` deep-copies the nominal
parameter set and injects an "lc_variants" section.  The engine reads
that section only (engine.py); with the section absent every flag is
off and the engine path is bit-for-bit the nominal one (test-enforced
by test_lc2_variants.py nominal byte-identity).

Zero fitting: the values are the registered ones, never tuned, swept
or selected.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
VARIANTS_CONFIG_PATH = HERE / "variants_lc2.json"

LC2_VARIANTS = ("nominal", "va", "vb", "vc", "vabc")

_DEFAULT_OFF = {"va_enabled": False, "vb_enabled": False,
                "vc_enabled": False}


def load_variant_config() -> dict:
    return json.loads(
        VARIANTS_CONFIG_PATH.read_text(encoding="utf-8"))


def variant_config_sha256() -> str:
    return hashlib.sha256(
        VARIANTS_CONFIG_PATH.read_bytes()).hexdigest()


def lc_variants_section(variant: str) -> dict:
    """The injected 'lc_variants' section for one named preset."""
    cfg = load_variant_config()
    if variant not in cfg["presets"]:
        raise ValueError(
            f"unknown LC2 variant {variant!r}; known: {sorted(cfg['presets'])}")
    sec = dict(_DEFAULT_OFF)
    sec.update(cfg["presets"][variant])
    vals = cfg["registered_values"]
    if sec.get("va_enabled"):
        sec["T_persist_s"] = float(vals["va_persistence"]["T_persist_s"])
    if sec.get("vb_enabled"):
        sec["T_lc_s"] = float(vals["vb_manoeuvre"]["T_lc_s"])
    if sec.get("vc_enabled"):
        sec["vc_threshold_zero"] = \
            bool(vals["vc_early_merge"]["threshold_zero"])
        sec["b_merge_mps2"] = float(vals["vc_early_merge"]["b_merge_mps2"])
    return sec


def variant_params(params: Mapping, variant: str) -> dict:
    """Deep-copied parameter set with the variant's flags injected under
    'lc_variants'.  The input mapping is never modified.  The section is
    ALWAYS injected (also for the 'nominal' preset, with every flag
    False) so the engine's OFF path is exercised identically."""
    p = copy.deepcopy(dict(params))
    p["lc_variants"] = lc_variants_section(variant)
    return p


def enabled_flags(params: Mapping) -> dict:
    """Which LC2 flags a parameter set carries (report provenance)."""
    sec = dict(params.get("lc_variants", {}) or {})
    return {k: bool(sec.get(k, False))
            for k in ("va_enabled", "vb_enabled", "vc_enabled")}
