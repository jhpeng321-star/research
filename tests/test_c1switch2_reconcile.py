"""Accounting must not convert an unreachable host into zero spent budget."""
from tools.c1switch2.reconcile import current_status


def snapshot():
    return dict(unix=1, hosts=dict(gpuR=dict(cpu=dict(wm=dict(charged_sim_s=2700,
        closed_actual_sim_s=2700)), gpu={})), alerts=[], observer={},
        bb_startup_stop={}, final_checkpoint_receipts={})


def test_missing_host_keeps_exact_totals_and_remaining_budget_unknown():
    status = current_status(snapshot(), 'receipt.json')
    assert status['charged_sim_s_at_latest_receipts'] is None
    assert status['registered_base_remaining_sim_s'] is None
    assert status['closed_actual_sim_s_at_latest_receipts'] is None
    assert status['charged_verified_lower_bound_sim_s'] == 7380
    assert status['new_gpu_s'] is None


def test_closed_actual_excludes_open_jobs_and_gpu_idle_is_retained():
    data = snapshot()
    data['hosts']['cpuQ'] = dict(cpu=dict(fleet_ppo=dict(charged_sim_s=1200,
        closed_actual_sim_s=0)), gpu={})
    data['hosts']['gpuA'] = dict(cpu={}, gpu={})
    for host in ('gpuA', 'gpuR'):
        data['hosts'][host]['gpu'] = dict(wm=dict(finished=dict(elapsed_s=100)),
                                          bb=dict(finished=dict(elapsed_s=20)))
    status = current_status(data, 'receipt.json')
    assert status['charged_sim_s_at_latest_receipts'] == 8580
    assert status['closed_actual_sim_s_at_latest_receipts'] == 7380
    assert status['registered_base_remaining_sim_s'] == 468380-8580
    assert status['new_gpu_s'] == 240


def test_live_continuation_keeps_original_debit_and_unknown_final_total():
    data = snapshot()
    data['hosts']['cpuQ'] = dict(cpu={},gpu={})
    data['hosts']['gpuA'] = dict(cpu={},gpu={})
    for host in ('gpuA','gpuR'):
        data['hosts'][host]['gpu'] = dict(wm=dict(finished=dict(elapsed_s=100)),
            bb_startup=dict(finished=dict(elapsed_s=20)),
            bb=dict(started=dict(unix=1),finished=None))
    status = current_status(data,'receipt.json')
    assert status['new_gpu_s'] is None
    assert status['finalized_gpu_s'] == 240
    for host in ('gpuA','gpuR'):
        data['hosts'][host]['gpu']['bb']['finished'] = dict(elapsed_s=30)
    assert current_status(data,'receipt.json')['new_gpu_s'] == 300
