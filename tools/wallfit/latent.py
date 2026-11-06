"""WM latent belief per observable row, from the FROZEN world model (gpuA runtime; zero SUMO).

For each WM query (record + causal 60 s observable history, exactly what the probe/mining WM
ranker consumed) the query is conditioned the same way the ranker conditions it
(tools.coopmwm.plan_service.request_memory, memory_mode 'warm'), and the frozen driver encoder is
read once: per row (slot order == record['rows'] order)
    embedding [K, D]  fused driver-encoder embedding
    values    [K, 9]  inferred driver parameters
    uncertainty [K, 9]
No weights change (torch.no_grad, eval()), no rollout, no SUMO.

CLI (gpuA):  python -m tools.wallfit latent --queries QUERIES.jsonl.gz --out LATENT.pt
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

from . import guard

SCHEMA = 'wallfit-latent-v1'
HISTORY_KEYS = ('history', 'history_mask', 'driver_history', 'driver_history_mask')


def _encoder_outputs(model, state, hist):
    """Frozen driver-encoder outputs; falls back to the encoder module when encode_drivers skips
    encoding for this rollout mode (same inputs _encode_drivers would build)."""
    import torch
    encoded, embedding, _ = model.encode_drivers(state, **hist)
    if encoded is not None:
        return encoded['embedding'], encoded['values'], encoded['uncertainty']
    enc = getattr(model, 'driver_encoder', None)
    if enc is None:
        raise ValueError('WM exposes no driver encoder latent')
    from wms.physics.state import state_kin, state_present
    kin = state_kin(state).float()
    present = state_present(state, kin)
    b, slots = kin.shape[:2]
    observed = hist.get('driver_history', hist.get('history'))
    mask = hist.get('driver_history_mask', hist.get('history_mask'))
    if observed is None or mask is None:
        observed = torch.zeros((b, slots, model.history_len, model.history_features), device=kin.device)
        mask = torch.zeros((b, slots, model.history_len), dtype=torch.bool, device=kin.device)
    out = enc(observed, mask.bool(), model._driver_interaction(kin, present),
              model._default_driver_parameters(kin, b, slots))
    return out['embedding'], out['values'], out['uncertainty']


def extract(rt, query, request_memory=None, diagnostic_source=None):
    """One query -> dict(vids, embedding [K,D], values [K,9], uncertainty [K,9]) as float32 tensors."""
    import torch
    record = copy.deepcopy(query['record'])
    if request_memory is None:
        from tools.coopmwm.plan_service import request_memory
    memory = request_memory(rt, record, {'memory_mode': 'warm'})
    if memory is not None:
        source = memory.source
    else:
        if diagnostic_source is None:
            from tools.wmteach.runtime import diagnostic_source
        source = diagnostic_source(rt, record, 0)
    hist = {k: v for k, v in (source.rollout_kwargs or {}).items() if k in HISTORY_KEYS and v is not None}
    model = rt.model
    was_training = getattr(model, 'training', False)
    if hasattr(model, 'eval'):
        model.eval()
    try:
        with torch.no_grad():
            emb, values, unc = _encoder_outputs(model, source.state, hist)
    finally:
        if was_training and hasattr(model, 'train'):
            model.train()
    n = len(record['rows'])
    if emb.shape[1] < n:
        raise ValueError('encoder slots fewer than observable rows')
    return dict(vids=[str(r['vid']) for r in record['rows']], embedding=emb[0, :n].float().cpu().clone(),
                values=values[0, :n].float().cpu().clone(), uncertainty=unc[0, :n].float().cpu().clone())


def apply_binding(path):
    """Apply worker_env.variables of a qualified GPU_BINDING.json to os.environ; returns the keys."""
    if not path:
        return {}
    binding = json.loads(guard.check_path(path).read_text())
    variables = {str(k): str(v) for k, v in ((binding.get('worker_env') or {}).get('variables') or {}).items()}
    os.environ.update(variables)
    return variables


def extract_all(rt, queries, limit=None, **kw):
    items, errors = {}, {}
    for i, (qid, d) in enumerate(sorted(queries.items())):
        if limit is not None and i >= limit:
            break
        try:
            items[qid] = extract(rt, d['query'], **kw)
        except Exception as e:      # recorded, never imputed
            errors[qid] = repr(e)
    return items, errors


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wallfit latent', description=__doc__.split('\n')[0])
    ap.add_argument('--queries', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fraction', type=float, default=.4, help='torch.cuda per-process memory fraction')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--binding', default=None,
                    help='GPU_BINDING.json of the qualified WM worker: its worker_env variables are applied '
                         'before bootstrap (same environment as the probe/mining WM rankers)')
    a = ap.parse_args(argv)
    import torch
    from .dataset import load_queries
    out = guard.check_path(a.out)
    queries = load_queries([guard.check_path(q) for q in a.queries])
    begin = time.time()
    applied = apply_binding(a.binding)
    from tools.wmteach.runtime import bootstrap
    rt = bootstrap()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.fraction)
    items, errors = extract_all(rt, queries, a.limit)
    payload = dict(schema=SCHEMA, items=items, errors=errors, n_queries=len(queries))
    tmp = out.with_name(out.name + '.tmp')
    torch.save(payload, str(tmp))
    os.replace(tmp, out)
    dims = sorted({int(v['embedding'].shape[-1]) for v in items.values()})
    receipt = dict(schema=SCHEMA, queries=[str(q) for q in a.queries], n_queries=len(queries), extracted=len(items),
                   binding=a.binding, binding_env_keys=sorted(applied),
                   errors=errors, embedding_dims=dims, wall_s=time.time() - begin, new_sumo_sim_s=0,
                   wm_weights_changed=False, out=str(out))
    Path(str(out) + '.receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in ('n_queries', 'extracted', 'embedding_dims', 'wall_s')}))
    if not items:
        raise SystemExit('no latent extracted: ' + json.dumps(errors)[:2000])
    return receipt
