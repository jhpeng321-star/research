import json
from unittest.mock import patch

from . import legacy_resource_pause_ops as pause


def test_registered_remote_pause_program_compiles_without_dispatch(tmp_path):
    captured=[]
    def remote(host,code,timeout):
        assert host=='cpuQ'
        compile(code,'<registered idle-boundary pause>','exec')
        captured.append(code)
        return json.dumps(dict(new_sim_s=0,updates={'B':21,'R':20}))
    with patch.object(pause,'register',return_value=({'line':'registered'},'a'*64)), \
            patch.object(pause,'remote',remote),patch.object(pause.ops,'EVIDENCE',tmp_path):
        pause.pause()
    assert len(captured)==1
    receipt=json.loads((tmp_path/'LEGACY_RESOURCE_PAUSE_2158B.json').read_text())
    assert receipt['updates']=={'B':21,'R':20} and receipt['new_sim_s']==0
