"""N1: wz3to2 scenario generation is bit-identical to the newmap probe / NEWMAP2 generators.

The generators are vendored verbatim under ``fixtures/newmap_generators`` (SHA256 pinned below;
originals: runs/active/C4-V7FT-20261017/newmap_probe/probe_core.py and newmap2/code/nm2_core.py).
When the originals exist on this machine they are compared too. No SUMO is started.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tools.coopmwm import map_descriptor as md, map_scenarios as ms

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MAP_DIR = REPO/'assets/research/newmap_candidates/wz3to2'
GEN = HERE/'fixtures'/'newmap_generators'
# LF-normalised SHA256 (raw on-disk originals: probe_core 749a8721..., population 6af5ee23..., nm2_core ad6b4bd6...).
PINNED = {'probe/probe_core.py': '867491861846937885e4266ea38de6149ffd07308c85ca9ddeef3a38924dc4e1',
          'probe/population.py': '5e8c61a6c1d5195e2830a5f6825e7a0ed4fc0c3260e243880ab956ddbff5849b',
          'nm2/nm2_core.py': 'ad6b4bd63426ca41f6ebf8b974db48eea508532f076981f652139de15fa20abb',
          'nm2/population.py': '5e8c61a6c1d5195e2830a5f6825e7a0ed4fc0c3260e243880ab956ddbff5849b'}
ORIGINALS = {'probe/probe_core.py': Path('E:/research-dense/runs/active/C4-V7FT-20261017/newmap_probe/probe_core.py'),
             'nm2/nm2_core.py': Path('E:/research-dense/runs/active/C4-V7FT-20261017/newmap2/code/nm2_core.py')}
CASES = [(989200, 2800, 0.4), (989201, 3200, 0.6), (989202, 2400, 0.2), (989377, 2800, 0.2), (989650, 3200, 0.4)]


def _norm_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _load(rel, name):
    path = GEN/rel
    saved_path, saved_pop = list(sys.path), sys.modules.pop('population', None)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.pop('population', None)
        if saved_pop is not None:
            sys.modules['population'] = saved_pop


def test_vendored_generators_are_the_pinned_originals():
    for rel, sha in PINNED.items():
        assert _norm_sha(GEN/rel) == sha, rel
    from traffic_shepherd.c4 import population
    assert _norm_sha(population.__file__) == PINNED['probe/population.py']
    for rel, path in ORIGINALS.items():
        if path.exists():
            assert _norm_sha(path) == PINNED[rel], rel


def _spec():
    return json.loads((MAP_DIR/'MAP_SPEC.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('seed,demand,share', CASES)
def test_routes_and_config_are_bit_identical_to_probe_and_nm2(tmp_path, seed, demand, share):
    probe, nm2 = _load('probe/probe_core.py', 'n1_probe_core'), _load('nm2/nm2_core.py', 'n1_nm2_core')
    spec = _spec()
    a, ra = probe.write_scenario(tmp_path/'probe', MAP_DIR, spec, 'wz3to2', seed, {'main': demand}, share)
    b, rb = nm2.write_scenario(tmp_path/'nm2', MAP_DIR, spec, seed, {'main': demand}, share)
    c, rc = ms.write_scenario(tmp_path/'n1', MAP_DIR, spec, 'wz3to2', seed, {'main': demand}, share)
    for name in ('requests.rou.xml', 'scenario.sumocfg'):
        ref = (tmp_path/'probe'/name).read_bytes()
        assert (tmp_path/'nm2'/name).read_bytes() == ref
        assert (tmp_path/'n1'/name).read_bytes() == ref
    assert ra == rb == rc and len(ra) > 500


@pytest.mark.parametrize('seed,demand,share', CASES[:3])
def test_env_builder_writes_the_probe_files_and_ledger_requests(tmp_path, seed, demand, share):
    probe = _load('probe/probe_core.py', 'n1_probe_core2')
    desc = md.load(MAP_DIR)
    scene, trips, manifest = ms.scenario_builder(desc)(tmp_path/'env', seed, demand, penetration=share,
                                                       role='exposed', population='textbook')
    _, reqs = probe.write_scenario(tmp_path/'probe', MAP_DIR, _spec(), 'wz3to2', seed, {'main': demand}, share)
    for name in ('requests.rou.xml', 'scenario.sumocfg'):
        assert (tmp_path/'env'/name).read_bytes() == (tmp_path/'probe'/name).read_bytes()
    assert scene == tmp_path/'env'/'scenario.sumocfg'
    assert [(t.vehicle_id, t.depart_s, t.is_cav) for t in trips] == [(r['id'], r['depart'], r['cav']) for r in reqs]
    assert all(t.driver_type == ('cav_typical' if t.is_cav else 'tb.'+t.vehicle_id) for t in trips)
    assert manifest['episode_seconds'] == 1200 and manifest['control_window_s'] == [300, 1200]
    assert manifest['driver_truth_file'] is None and not (tmp_path/'env'/'hetdata_truth.csv').exists()
    assert manifest['network_sha256'] == desc.network_sha256 and manifest['cell_key'] == ms.cell_key(demand, share)
    with pytest.raises(ValueError):   # never overwrite a scenario
        ms.scenario_builder(desc)(tmp_path/'env', seed, demand, penetration=share)


def test_builder_refuses_unregistered_cases(tmp_path):
    desc = md.load(MAP_DIR)
    build = ms.scenario_builder(desc)
    for seed, demand, share, pop in ((964001, 2800, .4, 'textbook'), (962100, 2800, .4, 'textbook'),
                                     (977010, 2800, .4, 'textbook'), (989150, 2800, .4, 'textbook'),  # 989000-989099 TRAIN since MAIN ~16:45
                                     (989200, 2700, .4, 'textbook'), (989200, 2800, .3, 'textbook'),
                                     (989200, 2800, .4, 'legacy3')):
        with pytest.raises(ValueError):
            build(tmp_path/f'{seed}_{demand}_{share}_{pop}', seed, demand, penetration=share, population=pop)


def test_newmap2_moments_map_one_to_one_onto_registered_worlds():
    nm2_seeds, nm2_demands, nm2_shares = (989200, 989201, 989202), (2400, 2800, 3200), (0.2, 0.4, 0.6)
    keys = set()
    for s in nm2_seeds:
        for d in nm2_demands:
            for p in nm2_shares:
                by_values = ms.world_for_moment(dict(seed=s, demand={'main': d}, penetration=p, t0=450))
                by_cell = ms.world_for_moment(dict(seed=s, cell=f'D{d}_p{int(round(p*100)):02d}', t0=800))
                assert by_values == by_cell == ms.guard(by_values, 'train')
                assert by_values in ms.worlds('train', seeds=[s])
                keys.add((by_values['seed'], by_values['cell']))
    assert len(keys) == 27
    assert len(ms.worlds('train')) == 400*9 and len(ms.worlds('final_test')) == 100*9


def test_registry_roles_ranges_and_aliasing():
    w = ms.world(989600, 2800, 0.4)
    assert ms.guard(w, 'final_test') == w
    with pytest.raises(ValueError):
        ms.guard(w, 'train')                      # held-out seeds never train
    t = ms.world(989300, 2400, 0.6)
    with pytest.raises(ValueError):
        ms.guard(t, 'final_test')                 # train seeds are not the final panel
    merge_world = dict(seed=989600, cell=0, repeat=0, penetration=.05, demand=1900)
    with pytest.raises(ValueError):
        ms.guard(merge_world, 'final_test')       # merge-map registry never aliases
    with pytest.raises(ValueError):
        ms.guard(dict(t, demand=2400.0), 'train')
    with pytest.raises(ValueError):
        ms.worlds('final_test', seeds=[989599])
    for lo, hi in ((964000, 964199), (962000, 962999), (977000, 977999)):
        assert ms.seed_scope(lo) is None and ms.seed_scope(hi) is None
    rec = ms.registry_record()
    assert rec['episode_s'] == 1200 and rec['warmup_s'] == 300 and rec['horizon_s'] == 901


def test_canonical_cell_and_n6_adapter_api():
    from tools.coopmwm import map_api
    assert map_api.CANONICAL_CELL_KEY == 'D3200_p60' and 3200 in ms.DEMANDS and 0.6 in ms.SHARES
    w = map_api.world(989650, 'D3200_p60')
    assert w == ms.world(989650, 3200, 0.6) == map_api.check_world(w, 'final_test')
    assert w['cell_key'] == 'D3200_p60' and w['map'] == 'wz3to2'
    assert map_api.seed_ranges() == dict(train=[989200, 989599], heldout=[989600, 989699],
                                         forbidden=[[964000, 964199], [962000, 962999], [977000, 977999]])
    panel = map_api.final_worlds()
    assert len(panel) == 100 and {x['cell_key'] for x in panel} == {'D3200_p60'}
    assert [x['seed'] for x in map_api.train_worlds(seeds=[989200, 989201])] == [989200, 989201]
    with pytest.raises(ValueError):
        map_api.check_world(map_api.world(989300), 'final_test')
    with pytest.raises(ValueError):
        map_api.world(989300, 'D3000_p60')
    assert map_api.descriptor(MAP_DIR).network_file_sha256.startswith('b6277898')
