"""Exercise the installed collector adapter through the actual fresh-module factory."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_collector_override_survives_factory_call():
    source = Path(__file__).resolve().parents[2]/'tools/autonomy/remote.py'
    tree = ast.parse(source.read_text())
    drive = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'drive')
    scope = next(n for n in ast.walk(drive) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'fast' for t in n.targets))
    body = next(n.body for n in ast.walk(drive) if isinstance(n, ast.With) and scope in n.body)
    start = body.index(scope)
    statements = body[start:start+3]
    loop = SimpleNamespace(fast_module=lambda: SimpleNamespace(collect=lambda *args: 'original'))
    env = {'loop': loop, 'collect': lambda *args: 'adapted', 'preflight': None, 'status': None, 'priority': None}
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<dispatch-regression>', 'exec'), env)
    assert loop.fast_module().collect(None, None, None, None, None) == 'adapted'
