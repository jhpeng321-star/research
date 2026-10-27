"""Canary-completion-first durable collection; dead spot boxes pause or use peer.

Uses the registered case expansion, per-world runner, artifact validation and
scientific smoke guard. No refund, seed replacement or different gate.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def collect(loop, cfg, state, ph, root, preflight, status, priority):
    fast = loop.fast_module()
    k = state['cycles'][-1]['cycle']
    reg = loop.check_registration(cfg)
    wm = state.get('policy_wm', state['accepted_wm'])
    journal = root / f'worlds_c{k:02d}.json'
    if journal.exists():
        doc = loop.read_json(journal)
    else:
        seeds = loop.allocate_seeds(cfg, state, reg, root)
        doc = {'cases': fast.cases(cfg, seeds), 'worlds': {}, 'attempts': {}, 'charged': 0,
               'policy_sha256': state['policy']['sha256'], 'wm_sha256': wm['sha256']}
        loop.write_atomic(journal, doc)
        loop.append(root / 'seed_ledger.jsonl', {'cycle': k, 'seeds': seeds, 'event': 'allocate_worlds'})
        state['seeds_used'] += seeds
        loop.save_state(cfg, state)
    if doc['policy_sha256'] != state['policy']['sha256'] or doc['wm_sha256'] != wm['sha256']:
        raise loop.Refused('collection provenance mismatch')
    ledger_path = root / 'seed_ledger.jsonl'
    ledger = [json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()]
    current = ledger[int(state.get('resumeprep_prior_ledger_rows', 0)):]
    doc['charged'] = sum(x.get('sim_s_charged', 0) for x in current if x.get('cycle') == k
                         and x.get('event') != 'smoke_reservation')
    state['sim_s_charged'] = max(state['sim_s_charged'], sum(x.get('sim_s_charged', 0) for x in ledger))
    for row in ledger:
        if row.get('event') == 'world_attempt' and row['cycle'] == k:
            key = str(row['seed'])
            doc['attempts'][key] = max(doc['attempts'].get(key, 0), row['attempt'])
    long_used = sum(x.get('long_credit_sim_s_charged', x.get('long_group_sim_s_charged', 0))
                    for x in current if x.get('cycle') == k)
    per = loop.collect_sim_s(cfg) // cfg['collect']['worlds_per_cycle']
    long_per = loop.long_group_sim_s(cfg) // cfg['collect']['worlds_per_cycle']
    pending = [x for x in doc['cases'] if not (
        str(x['seed']) in doc['worlds'] and fast.artifact_valid(doc['worlds'][str(x['seed'])]['artifact']))]
    # Budget the WHOLE remaining cohort before even the first canary. Retries retain charges.
    if state['sim_s_charged'] + len(pending) * per > reg['budget']['sim_s_total']:
        raise loop.Refused('WAIT_OWNER_BUDGET: remaining collection would exceed total cap')
    if doc['charged'] + len(pending) * per > reg['budget']['cycle_attempt_allowances'][str(k)]:
        raise loop.Refused('WAIT_OWNER: cycle attempt allowance exhausted')
    if long_used + len(pending) * long_per > reg['collection']['cycle_long_attempt_allowances'][str(k)]:
        raise loop.Refused('WAIT_OWNER: included long attempt allowance exhausted')
    live_boxes = preflight(cfg, state, doc['cases'])
    if not live_boxes:
        raise RuntimeError('SPOT_UNAVAILABLE: no collector passed zero-charge preflight')
    smoke_seeds = {x['seed'] for x in doc['cases'][:4]}
    if doc.get('native_smoke') and fast.smoke_stops(cfg, k, doc['native_smoke']):
        raise loop.Refused('HARMFUL registered native smoke; no retry')

    def charge(item, box):
        nonlocal long_used
        priority()
        key = str(item['seed'])
        attempt = doc['attempts'].get(key, 0) + 1
        row = {'cycle': k, 'seed': item['seed'], 'host': box, 'attempt': attempt,
               'sim_s_charged': per, 'event': 'world_attempt',
               'long_credit_sim_s_charged': long_per,
               'long_group_sim_s_charged': 300 if cfg['collect'].get('long_group', {}).get('enabled') else 0,
               'long_single_sim_s_charged': 300 if cfg['collect'].get('long_single_hold', {}).get('enabled') else 0}
        loop.append(ledger_path, row)
        doc['attempts'][key] = attempt
        doc['charged'] += per
        state['sim_s_charged'] += per
        long_used += long_per
        loop.save_state(cfg, state)
        loop.write_atomic(journal, doc)
        return attempt

    def complete(item, receipt):
        doc['worlds'][str(item['seed'])] = receipt
        loop.write_atomic(journal, doc)
        incidents = sum((w['collect'].get('branch_collisions') or 0) +
                        (w['collect'].get('branch_teleports') or 0) for w in doc['worlds'].values())
        if incidents > reg['stop'].get('max_branch_incidents_per_cycle', 0):
            state['stopped'] = 'SUMO safety incident in completed world'
            loop.save_state(cfg, state)
            raise loop.Refused(state['stopped'])

    canary_done = set()
    for batch in ([x for x in pending if x['seed'] in smoke_seeds],
                  [x for x in pending if x['seed'] not in smoke_seeds]):
        ordered = list(batch)
        # Full valid completion is a stronger startup barrier than partial frame proof.
        for box in live_boxes:
            if box in canary_done or not ordered:
                continue
            item = next((x for x in ordered if x['box'] == box), ordered[0])
            ordered.remove(item)
            status('CANARY_RUNNING', box=box, seed=item['seed'])
            attempt = charge(item, box)
            complete(item, fast._world(loop, cfg, state, ph, root, item, box, attempt))
            canary_done.add(box)
            loop.append(root / 'cycle_ledger.jsonl', {'event': 'AUTONOMY_canary_completed',
                        'cycle': k, 'box': box, 'seed': item['seed'], 'attempt': attempt})
        failures = []
        from contextlib import ExitStack
        with ExitStack() as stack:
            pools = {box: stack.enter_context(ThreadPoolExecutor(
                max_workers=int(cfg['fastloop']['workers_per_box']))) for box in live_boxes}
            futures = {}
            for item in ordered:
                box = item['box'] if item['box'] in live_boxes else live_boxes[item['index'] % len(live_boxes)]
                attempt = charge(item, box)
                futures[pools[box].submit(fast._world, loop, cfg, state, ph, root, item, box, attempt)] = item
            for future in as_completed(futures):
                item = futures[future]
                try:
                    complete(item, future.result())
                except loop.Refused:
                    raise
                except Exception as error:
                    failures.append(f'{item["seed"]}: {error!r}')
        if failures:
            raise RuntimeError('worlds incomplete; charged peer retry: ' + '; '.join(failures))
        if all(str(x['seed']) in doc['worlds'] for x in doc['cases'][:4]):
            from wms.control.nativeinit import artifact_world_delta, smoke_guard
            doc['native_smoke'] = smoke_guard([artifact_world_delta(
                doc['worlds'][str(x['seed'])]['artifact'], int(cfg['collect']['branch_points']))
                for x in doc['cases'][:4]])
            loop.write_atomic(journal, doc)
            if fast.smoke_stops(cfg, k, doc['native_smoke']):
                state['stopped'] = 'native smoke harmful'
                loop.save_state(cfg, state)
                raise loop.Refused(state['stopped'])
    worlds = [doc['worlds'][str(x['seed'])] for x in doc['cases']]
    return {'journal': str(journal), 'worlds': worlds, 'seeds': [x['seed'] for x in doc['cases']],
            'sim_s_charged': doc['charged'],
            'branch_collisions': sum(w['collect'].get('branch_collisions') or 0 for w in worlds),
            'branch_teleports': sum(w['collect'].get('branch_teleports') or 0 for w in worlds)}
