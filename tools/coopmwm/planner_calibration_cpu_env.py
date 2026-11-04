"""Explicit pytest-only restoration of the checkout's missing selection dependency.

Use ``-p tools.coopmwm.planner_calibration_cpu_env``. The existing rehearsal stores
the verbatim observable features implementation from tools.leverprobe.selection.
Load ONLY that literal using AST: no rehearsal imports, backend replacement or
test suppression. An installed real dependency always takes precedence.
"""
import ast
import importlib
from pathlib import Path
import sys
import types


def selection_module():
    source = Path(__file__).with_name('rehearsal_cpu.py')
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_SELECTION_FEATURES' for t in node.targets)):
            literal = ast.literal_eval(node.value)
            module = types.ModuleType('tools.leverprobe.selection')
            module.__file__ = str(source)
            exec(compile(literal, str(source)+'::_SELECTION_FEATURES', 'exec'), module.__dict__)
            return module
    raise RuntimeError('registered observable selection source not available')


def pytest_configure(config):
    try:
        importlib.import_module('tools.leverprobe.selection')
        return
    except ModuleNotFoundError as exc:
        if exc.name not in ('tools.leverprobe', 'tools.leverprobe.selection'):
            raise
    created = {}
    if 'tools.leverprobe' not in sys.modules:
        package = types.ModuleType('tools.leverprobe')
        package.__path__ = []
        created['tools.leverprobe'] = package
        sys.modules['tools.leverprobe'] = package
    module = selection_module()
    sys.modules['tools.leverprobe.selection'] = module
    sys.modules['tools.leverprobe'].selection = module
    created['tools.leverprobe.selection'] = module
    config._t1c_selection_modules = created


def pytest_unconfigure(config):
    created = getattr(config, '_t1c_selection_modules', {})
    package = sys.modules.get('tools.leverprobe')
    if package is not None and getattr(package, 'selection', None) is created.get('tools.leverprobe.selection'):
        del package.selection
    for name, module in created.items():
        if sys.modules.get(name) is module:
            del sys.modules[name]
