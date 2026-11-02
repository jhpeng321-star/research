from .ls_fast_ops import roll_target


TIER=(('gpuA','a'),('gpuR','r'))


def worker(root,fast=True,ready=True,state='R',failure=None):
    return dict(owner=dict(root=root,generation='FASTK2V5' if fast else 'VALIDITY2'),
        ready=ready,state=state,failure=failure)


def test_existing_qualifier_blocks_another_gpu_upgrade():
    g=dict(gpuA=dict(workers=[worker('a',ready=False)]),
           gpuR=dict(workers=[worker('r',fast=False)]))
    assert roll_target(TIER,g) is None


def test_completed_first_slot_allows_only_next_slot():
    g=dict(gpuA=dict(workers=[worker('a')]),gpuR=dict(workers=[worker('r',fast=False)]))
    assert roll_target(TIER,g)==('gpuR','r')
    g['gpuR']['workers']=[worker('r')]
    assert roll_target(TIER,g) is None


def test_dead_ready_receipt_cannot_complete_rollout():
    g=dict(gpuA=dict(workers=[worker('a',state='exited')]),gpuR=dict(workers=[worker('r')]))
    assert roll_target(TIER,g)==('gpuA','a')
