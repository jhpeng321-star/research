"""Run only new orchestration CPU tests without touching the active runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--receipt', default='pilot_cpu_2230.json')
    a = p.parse_args()
    names = ('pilot_handoff', 'pilot_recipe', 'pilot_continue', 'pilot_continue_supervise',
             'pilot_handoff_test', 'pilot_recipe_test', 'pilot_continue_test')
    directory = Path(__file__).resolve().parent
    sources = {name: (directory / (name + '.py')).read_text(encoding='utf-8') for name in names}
    code = 'import sys, types, unittest\n'
    for name, source in sources.items():
        full = 'tools.candB.' + name
        code += (f'm = types.ModuleType({full!r}); m.__file__ = {name + ".py"!r}\n'
                 f'sys.modules[{full!r}] = m\nexec(compile({source!r}, m.__file__, "exec"), m.__dict__)\n')
    code += ('suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[n]) for n in '
             f'{["tools.candB." + n for n in names if n.endswith("_test")]!r})\n'
             'result=unittest.TextTestRunner(verbosity=2).run(suite)\n'
             'raise SystemExit(not result.wasSuccessful())\n')
    result = subprocess.run(['ssh', 'gpuA', 'env', '-C', a.runtime, 'CUDA_VISIBLE_DEVICES=',
                             'PYTHONPATH=src:.', a.python, '-B', '-'], input=code,
                            text=True, capture_output=True, timeout=180)
    receipt = dict(label='EXPLORATORY', cuda_visible_devices='', returncode=result.returncode,
                   sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
                   stdout=result.stdout, stderr=result.stderr)
    with (directory / a.receipt).open('x') as stream:
        json.dump(receipt, stream, indent=1)
        stream.write('\n')
    print(result.stdout + result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
