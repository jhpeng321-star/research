"""Read only capacity at known permitted roots; never traverse research data."""
import json
from pathlib import Path
import shutil

paths = ['/root', '/root/autodl-tmp', '/dev/shm', '/root/coopaudit20261031',
         '/root/sumocf20261031', '/root/sumocf20261031/traces']
print(json.dumps({p:dict(realpath=str(Path(p).resolve()), free=shutil.disk_usage(p).free,
                        total=shutil.disk_usage(p).total) for p in paths}, indent=1))
