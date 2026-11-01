"""Run new orchestration against an explicitly frozen, untouched model runtime.

The versioned runner is loaded by absolute path while every model/recipe import
comes from --runtime-root. This lets extended WM2/v4 forecasts retain their
original physics and prevents a merged v5 module from entering a v4 baseline.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', choices=('WM2', 'FULL'), required=True)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--wm-config', required=True)
    parser.add_argument('--wm', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--registered-panel')
    parser.add_argument('--development-direct', action='store_true', help='one baseline kernel replay on dev969420/future0 only')
    parser.add_argument('--baseline-audit')
    args, remaining = parser.parse_known_args()
    control = Path(__file__).resolve().parents[2]
    runtime = Path(args.runtime_root).resolve()
    if runtime == control:
        raise ValueError('orchestration and frozen model stages must be separate')
    sys.path[:0] = [str(runtime), str(runtime/'src'), str(control)]
    # New model stages also contain an older copy of tools.rb2int. Pin ONLY the
    # orchestration package to this control stage; physics/recipe packages still
    # resolve from the frozen model runtime. Otherwise v5 silently imports stale
    # adapters (or fails the subsequent source guard) despite correct sys.path.
    import tools
    package_name = 'tools.rb2int'
    package = sys.modules.get(package_name)
    if package is None:
        package_spec = importlib.util.spec_from_file_location(package_name,
            control/'tools/rb2int/__init__.py',
            submodule_search_locations=[str(control/'tools/rb2int')])
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package
        package_spec.loader.exec_module(package)
        setattr(tools, 'rb2int', package)
    elif Path(package.__file__).resolve().parents[2] != control:
        raise ValueError('orchestration package was imported from the frozen model stage')
    package.__path__ = [str(control/'tools/rb2int')]
    from tools.rbeval.core import digest, load, safe
    panel = None
    if args.development_direct:
        if args.registered_panel or not args.baseline_audit:
            raise ValueError('development baseline replay needs its audit and no final panel')
        audit = load(args.baseline_audit)
        expected = 'wm2' if audit['label'] == 'WM2' else 'WM-PHYS-FULL-v4-gpu'
        if (audit['schema'] != 'rb2int-frozen-baseline-audit/1' or not audit['passed'] or
                audit['label'] not in ('WM2', 'FULL-v4') or
                args.model_name != expected):
            raise ValueError('development replay requires the original frozen baseline')
        model_binding = dict(audit, sources=audit['semantic_frozen_sources'])
    else:
        if not args.registered_panel:
            raise ValueError('final evaluation requires an explicit released registered panel')
        from tools.rb2int.inputs import validate_panel
        panel = validate_panel(args.registered_panel)
        release = load(panel['release_path'])
        model_binding = release['models'][args.model_name]
    if (str(safe(model_binding['runtime_root']).resolve()) != str(runtime) or
            model_binding['config_sha256'] != digest(args.wm_config) or
            model_binding['wm_sha256'] != digest(args.wm) or
            model_binding['variant'] != args.variant):
        raise ValueError('frozen model/runtime/recipe binding differs')
    sources = model_binding['sources']
    if not sources:
        raise ValueError('frozen runtime needs a nonempty exact source manifest')
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError('frozen model source changed: '+str(path))
    # tools is a namespace package: the frozen runtime takes import precedence.
    # Relative .core imports below intentionally resolve to the frozen scorer
    # primitives, while this selected runner remains orchestration-only.
    runner_path = control/'tools/rbeval/run_wm.py'
    spec = importlib.util.spec_from_file_location('tools.rbeval._rb2_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    for key in ('tools.rb2int.inputs', 'tools.rb2int.evaluate'):
        module = sys.modules.get(key)
        if module is not None and Path(module.__file__).resolve().parents[2] != control:
            raise ValueError('orchestration import came from a different stage')
    identity_sources = [Path(__file__), control/'tools/rb2int/inputs.py',
                        *([Path(args.registered_panel), Path(panel['release_path']),
                           Path(panel['registration_path'])] if panel is not None else [Path(args.baseline_audit)])]
    identity_sources += [Path(p) for p in sources]
    sys.argv = [str(Path(__file__)), '--runtime-root', str(runtime), '--wm', args.wm,
        '--wm-config', args.wm_config, '--model-name', args.model_name, *remaining]
    if panel is not None:
        sys.argv += ['--registered-panel', args.registered_panel]
    selection = dict(worlds=(969420,), future_indices=(0,)) if args.development_direct else {}
    if args.variant == 'WM2':
        if args.model_name != 'wm2':
            raise ValueError('WM2 keeps its original model identity/checkpoint guard')
        runner.main(identity_sources=identity_sources, **selection)
    else:
        os.environ['WMPHYS_VARIANT'] = 'FULL'
        from tools.rbint.recipe import candidate_future, load_candidate
        if Path(sys.modules[load_candidate.__module__].__file__).resolve().parents[2] != runtime:
            raise ValueError('candidate recipe did not come from its frozen stage')
        runner.main(model_loader=load_candidate, future_builder=candidate_future,
                    identity_sources=identity_sources, **selection)


if __name__ == '__main__':
    main()
