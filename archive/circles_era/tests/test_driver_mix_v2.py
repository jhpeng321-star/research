"""Regression for the actual preallocation seed-role failure."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / 'output/driver_mix_v2/campaign.py'
spec = importlib.util.spec_from_file_location('oe43_v2_campaign', PATH)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_every_registered_job_uses_existing_seed_role_and_passes_production_gate():
    for seed in campaign.SEEDS:
        for arm in ('damping', 'h20'):
            job = campaign.base_job(seed, arm)
            assert job['role'] == 'development'
            campaign.pe.validate_job(job)


def test_correction_shares_original_budget_ledger_and_preserves_total_ceiling():
    assert campaign.LEDGER.name == 'oe43_driver_mix.json'
    assert campaign.CEILINGS['total_seconds'] == 9600
    assert campaign.CEILINGS['categories']['damping']['max_seconds'] == 4800
    assert campaign.CEILINGS['categories']['h20']['max_seconds'] == 4800
