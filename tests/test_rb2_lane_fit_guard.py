import pytest

from tools.rblane.fit_guard import guarded_start_source


def test_reserved_world_read_cannot_happen_in_registered_loader():
    source='''def make_starts_wmdata(model):
    worlds = [{'seed': 969420}, {'seed': 969432}, {'seed': 969421}]
    for i, world in enumerate(worlds):
        seed = world['seed']
        read_start(seed)
'''
    tree,_=guarded_start_source(source,[969420,969421])
    observed=[]
    def read_start(world):
        assert world!=969432, 'reserved read must never occur'
        observed.append(world)
    namespace=dict(read_start=read_start)
    exec(compile(tree,'[own synthetic loader]','exec'),namespace)
    namespace['make_starts_wmdata'](None)
    assert observed==[969420,969421]


def test_bad_selection_or_changed_loader_fails_closed():
    with pytest.raises(ValueError,match='development12'):
        guarded_start_source('pass',[969432])
    with pytest.raises(ValueError,match='world-loop'):
        guarded_start_source('def make_starts_wmdata(model):\n    read_all_starts()', [969420])
