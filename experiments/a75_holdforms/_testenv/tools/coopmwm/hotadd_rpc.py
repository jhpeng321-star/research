"""Same restricted spool RPC, exposing the two additional background gate receipts."""
import json
from pathlib import Path
import sys
from . import warm_rpc


def handle(root,value):
    result=warm_rpc.handle(root,value)
    status=result if value.get('action')=='status' else result.get('status')
    if status is not None:
        for name in ('STARTUP_COMMITTED_K4_H300.json','STARTUP_COMMITTED_K8_H420.json'):
            path=Path(root)/name
            if path.exists():status['receipts'][name]=json.loads(path.read_text())
    return result


if __name__=='__main__':
    data=sys.stdin.buffer.read(32*1024*1024+1)
    if len(data)>32*1024*1024:raise ValueError('oversized warm RPC')
    print(json.dumps(handle(Path(sys.argv[1]),json.loads(data)),allow_nan=False),flush=True)
