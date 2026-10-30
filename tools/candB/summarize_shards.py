"""Offline deadline report from all completed, paired measurement arms; no imputation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.distill.runtime import MAPPO_SHA, WM_SHA, check_metadata, safe_path, sha256
from tools.setupaudit.summarize2 import cluster_plan, estimate

METRICS = ('J90', 'J300', 'objective90_loss', 'objective300_loss')
ARMS = ('MAPPO_native_lanes', 'MAPPO_speed_partial', 'CIRCLES', 'PI')
LABELS = dict(native='native', CTRL0='CTRL0', MAPPO_native_lanes='MAPPO proxy aecd7d70',
    MAPPO_speed_partial='speed-only MAPPO partial i25', CIRCLES='CIRCLES damping', PI='PI external-U')


def stats(rows, metrics=METRICS):
    worlds = [r['metadata'][0]['world'] for r in rows]
    unique, plan = cluster_plan(worlds, draws=5000, seed=20261030)
    result = dict(n_starts=len(rows), n_worlds=len(unique))
    for metric in metrics:
        values = np.array([[np.mean(r['value'][metric]), np.mean(r['reference'][metric])]
                           for r in rows])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite completed arm')
        result[metric] = dict(value=estimate(values[:, 0], worlds, unique, plan),
            paired_delta=estimate(values[:, 0]-values[:, 1], worlds, unique, plan))
    return result


def read(path):
    return json.loads(safe_path(path).read_text())


def analyze(root):
    inputs = {}
    rows = {}
    contracts = []
    plans = set()
    for directory in ('circles', 'pi', 'mappo'):
        path = root/directory/'RAW.json'
        if not path.exists():
            continue
        doc = read(path)
        assert doc['measure_only'] and not doc['passed'] and not doc['complete']
        assert not doc['launch_authorized'] and doc['K'] == 4 and doc['qualified_K'] == 8
        assert doc['wm_sha256'] == WM_SHA and doc['mappo_sha256'] == MAPPO_SHA
        assert doc['mix_connected'] and sum(map(sum, doc['mix_assignment_receipts'])) > 0
        assert doc['gpu_memory_fraction'] <= .40
        contracts.append(doc['runtime_contract'])
        plans.add(doc['input_starts_sha256'])
        inputs[directory] = dict(path=path.as_posix(), sha256=sha256(path),
            config_sha256=doc['config_sha256'], wm_config_sha256=doc['wm_config_sha256'],
            K=doc['K'], source_manifest=doc['source_manifest'],
            mappo_speed_partial=doc.get('mappo_speed_partial'))
        for arm in doc['arms']:
            completed = []
            for index, row in enumerate(doc['rows']):
                if arm not in row['arms']:
                    continue
                check_metadata(row['metadata'])
                assert len(row['metadata']) == 1
                value, reference = row['arms'][arm], row['paired_CTRL0']
                for item in (value, reference):
                    assert len(item['J90']) == len(item['J300']) == 4
                    np.testing.assert_allclose(item['objective90_loss'], -item['dense_credit'])
                    np.testing.assert_allclose(item['objective300_loss'],
                        -item['dense_credit']-item['long_credit'], rtol=1e-6, atol=1e-4)
                completed.append(dict(index=index, metadata=row['metadata'], value=value,
                    reference=reference, native=row['paired_native_physical']))
            if completed:
                assert arm not in rows, 'duplicate completed arm inputs'
                rows[arm] = completed
    assert rows, 'no completed diagnostic arms yet'
    assert len(plans) == 1 and all(c == contracts[0] for c in contracts)
    # Availability-only canonical reference; never choose by observed controller outcomes.
    canonical = max(rows, key=lambda arm: (len(rows[arm]), arm))
    reference_rows = rows[canonical]
    summary = {arm: stats(items) for arm, items in rows.items()}
    summary['CTRL0'] = stats([{**r, 'value': r['reference']} for r in reference_rows])
    # Physical native is already the actual learned-native row in each WM image.
    # Native actor credit is a different counterfactual intervention, not inferred here.
    native_rows = [{**r, 'value': r['native']} for r in reference_rows]
    summary['native'] = stats(native_rows, ('J90', 'J300'))
    old_path = root.parent/'rankcheck_saved_k8/CANDB_RANKCHECK.json'
    old = read(old_path)
    old_row = old['rows'][0]
    native_actor = stats([dict(metadata=old_row['metadata'], value=old_row['arms']['native'],
                              reference=old_row['arms']['CTRL0'])])
    zero_path = root/'zero/RAW.json'
    zero = read(zero_path)
    assert zero['runtime_contract'] == contracts[0]
    assert zero['input_starts_sha256'] in plans
    assert zero['zero_identity']['bitwise'] and zero['zero_identity']['complete_300s']
    zero_row = zero['rows'][0]
    for metric in METRICS:
        assert zero_row['arms']['candB_zero'][metric] == zero_row['paired_CTRL0'][metric]
    common = set.intersection(*(set(r['index'] for r in rows[arm]) for arm in ARMS)) if all(
        arm in rows for arm in ARMS) else set()
    common_stats = {}
    cross_host_reference_max = 0.
    if common:
        for arm in ARMS:
            selected = [r for r in rows[arm] if r['index'] in common]
            for r in selected:
                reference = next(x for x in reference_rows if x['index'] == r['index'])
                assert r['metadata'] == reference['metadata'], 'unmatched TRAIN start'
                cross_host_reference_max = max(cross_host_reference_max, float(np.max(np.abs(
                    np.asarray(r['reference']['J300'])-np.asarray(reference['reference']['J300'])))))
            common_stats[arm] = stats(selected)
        selected = [r for r in reference_rows if r['index'] in common]
        common_stats['CTRL0'] = stats([{**r, 'value': r['reference']} for r in selected])
        common_stats['native'] = stats([{**r, 'value': r['native']}
                                       for r in selected], ('J90', 'J300'))
    proxy = summary.get('MAPPO_native_lanes')
    diagnostic_gate = bool(proxy and proxy['n_worlds'] >= 2
                          and proxy['J300']['paired_delta']['hi'] < 0)
    return dict(schema='candB_rankcheck_deadline/1', complete=False, passed=False,
        launch_authorized=False, K=4, qualified_K=8, input_starts_sha256=next(iter(plans)),
        inputs=inputs, canonical_reference_arm=canonical, per_arm=summary,
        native_actor_K8=native_actor, common_start_indices=sorted(common),
        common=common_stats, diagnostic_proxy_condition=diagnostic_gate,
        cross_host_CTRL0_max_abs_J300=cross_host_reference_max,
        full_R2_demonstrated=False, zero_identity=zero['zero_identity'],
        current_output_head_zero=zero['current_output_head_zero'],
        dense_horizons_s=sorted({r['value']['dense_horizon_s'] for values in rows.values()
                                 for r in values}),
        saved_K8_sha256=sha256(old_path), zero_sha256=sha256(zero_path))


def cell(stat, worlds, signed=False):
    sign = '+' if signed else ''
    value = f'{stat["mean"]:{sign}.1f}'
    return (value+' [NE]' if worlds < 2 else
            value+f' [{stat["lo"]:{sign}.1f},{stat["hi"]:{sign}.1f}]')


def report(result, final_sha, stamp):
    summaries = result['per_arm']
    def ranking(metric, *, actor=False):
        data = {k:v for k,v in result['common'].items() if not (actor and k == 'native')}
        return ' < '.join(LABELS[arm] for arm in sorted(
            data, key=lambda k: data[k][metric]['value']['mean'])) or 'incomplete common panel'
    def table(metrics, *, actor=False):
        lines = []
        for arm in ('native', 'CTRL0', *ARMS):
            if arm not in summaries:
                lines.append(f'| {LABELS[arm]} | pending | NE | NE | NE | NE |')
                continue
            s = result['native_actor_K8'] if actor and arm == 'native' else summaries[arm]
            k = 8 if actor and arm == 'native' else 4
            fields = [LABELS[arm], f'{s["n_starts"]}/{s["n_worlds"]}/K{k}']
            for metric in metrics:
                fields += [cell(s[metric]['value'], s['n_worlds']),
                           cell(s[metric]['paired_delta'], s['n_worlds'], True)]
            lines.append('| '+' | '.join(fields)+' |')
        lines.append('| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |')
        return lines
    first = next(iter(result['inputs'].values()))
    proxy = summaries.get('MAPPO_native_lanes')
    outcome = ('HOLDS in the reduced-K diagnostic only' if result['diagnostic_proxy_condition'] else
               'NOT DEMONSTRATED in available reduced-K data')
    lines = [
        '# S-RANKCHECK — deadline measurement (incomplete; no launch authority)',
        f'As of {stamp}; final verified code+evidence SHA `{final_sha}`.',
        'Report-delivery SHA: `git log -1 --format=%H -- docs/paper_results/RANKCHECK.md`.',
        'Measured source SHAs: `2dcb6af5` (PI/CIRCLES), `3cd6481e` (MAPPO/zero), `05f7bfba` (retained serial K8).',
        '## Contract and completeness',
        'WM2 `b8797242` + learned mix ON; candidate-B resolved config, CTRL0 law/reference, all vehicles + entrants + road deficit + source waiting; no training/SUMO.',
        'MAIN19:10 fallback: **K4 = first four original K8 CRN plans**, same dense-suffix/joint-q10 and gain-gated300s algebra; this is a reduced-K estimator, not K8 qualification.',
        f'Config SHA256 `{first["config_sha256"]}`; WM inference SHA256 `{first["wm_config_sha256"]}`.',
        f'Sixteen actual TRAIN-sampler starts exported; plan SHA256 `{result["input_starts_sha256"]}`. Completed counts appear below; no missing arm/future imputed.',
        'Raw JSON/argv/ownership: `tools/candB/evidence/rankcheck_parallel/{circles,pi,mappo,zero}/`; retained K8: `../rankcheck_saved_k8/CANDB_RANKCHECK.json`.',
        'cpuQ read-only first-inspection export: `~/mappospeed/production2/policy_00025.pt`, iteration25/169, **PARTIALLY TRAINED**, frozen before outcomes; SHA256 `69e414e53e7f520bbd112ff292aa207d3da6388d73c3ac91e894079dfe40a543`.',
        'Native lanes throughout; aecd7d70 proxy ignores its learned lane head. PI/CIRCLES use the published TimelineControllers; CPU/CUDA sensor-copy parity4/4 on BOTH hosts.',
        'CIs: paired equal-WORLD percentile bootstrap,5000 draws/seed20261030; average futures/starts inside each world first. NE=not estimable from one world.',
        '## Physical J (veh-s; lower is better)',
        'Each row uses ALL its completed starts; dJ is paired to that row\'s actual CTRL0 futures. Different completion counts mean absolute row means are not a balanced ranking.',
        '| Controller | starts/worlds/K | J90 [95% CI] | dJ90 vs CTRL0 | J300 [95% CI] | dJ300 vs CTRL0 |',
        '|---|---|---:|---:|---:|---:|',
        *table(('J90', 'J300')),
        f'Balanced common-start ranking ({len(result["common_start_indices"])} starts),90s: {ranking("J90")}.',
        f'Balanced common-start ranking,300s: {ranking("J300")}.',
        f'Cross-host CTRL0 reference max absolute future-J300 difference on common starts: {result["cross_host_CTRL0_max_abs_J300"]:.3f} veh-s; every dJ remains paired within its own process.',
        '## Actor objective (loss at scale1; lower is better)',
        f'L90 is TRAIN-resolved dense-prefix suffix/joint-q10 loss (observed caps {result["dense_horizons_s"]}s); L300 adds registered gain-gated accrued300s credit. These are not physical J.',
        '| Controller | starts/worlds/K | L90 [95% CI] | dL90 vs CTRL0 | L300 [95% CI] | dL300 vs CTRL0 |',
        '|---|---|---:|---:|---:|---:|',
        *table(('objective90_loss', 'objective300_loss'), actor=True),
        f'Balanced common-start actor ranking,dense: {ranking("objective90_loss", actor=True)}.',
        f'Balanced common-start actor ranking,dense+300s: {ranking("objective300_loss", actor=True)}.',
        'Native actor loss is the retained single-start K8 intervention; its larger-panel actor CI is missing. Native/CTRL0 physical rows are reused from actual reference rows, not extra rollouts.',
        'Retained serial K8 first start: proxy dJ90+4.43,dJ300+249.59; native dJ90+91.91,dJ300+559.59 veh-s (one world,CI NE). All original saved arms remain on disk.',
        'Fresh-zero B: one full300s scientific-state array equality PASS vs CTRL0, plus exact physical/actor equality. Current u191 head is nonzero; no construction-based current-A identity is claimed.',
        '## Existing SUMO evidence and R2',
        'CTRLARM600s: CTRL0-native **-1956.53 [-4086.66,+144.50]**; lane-on MAPPO-CTRL0 **-623.95 [-2044.25,+664.51]** veh-s; both include0. Exact source: `E:/research-dense/docs/paper_results/CTRLARM.md:9-17`.',
        'These lane-on/600s reused-world results are contextual, not matched native-lane/300s speed-only efficacy; the partial speed-only export has no matched real result here.',
        'No numeric PI/CIRCLES SUMO contrast in either named paper_results directory. `E:/research-dense/runs/active/C4-V5-20261014/REGISTRATION.md:96-99` registers damping/PI arms only; active C4-EVAL registrations are absent.',
        'SETUPAUDIT2 physical all-vehicle/source/mix scoring agrees, but its TRAIN export calls train.sample directly (tools/setupaudit/inspect_inputs.py:46-58), bypassing B headroom/tilt/exclusions; no actor loss is saved. Fixedupdate190/horizon600 slots/future-specific lane uniforms also differ. It cannot substitute for this exact B panel.',
        f'**R2 proxy clause: {outcome}. Full registered R2: NOT DEMONSTRATED (coverage incomplete; authorized K4 fallback disclosed); no matched real PI/CIRCLES/native-lane alternative establishes it.**',
        'Binding rule: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md:4951-4956`. Point-estimate order/overlapping intervals alone do not establish the alternative.',
        '## Execution and verification',
        'Fresh source staging via tools/candB/stage.sh; gpuA <=2x.19, gpuR .40; own-PID/incarnation controls only. Peer WMREQUAL/SETUPAUDIT jobs untouched; all failed/timing attempts retained.',
        '16 CPU tests pass (2 local CUDA skips);4 CPU/CUDA equivalence tests pass independently on each GPU host; Ruff/diff PASS. Paired CIs independently recomputed from every retained completed start.',
        'Every receipt remains measure_only=true,complete=false,passed=false,launch_authorized=false; historical mismatched CCT-2 remains unqualified. No push/deletions.',
    ]
    if proxy:
        lines.insert(-6, 'Available proxy dJ300: '+cell(proxy['J300']['paired_delta'],
            proxy['n_worlds'], True)+' veh-s; speed-only R2 needs upper95 below0.')
    assert len(lines) <= 70, len(lines)
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('tools/candB/evidence/rankcheck_parallel'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--final-sha')
    parser.add_argument('--stamp', default='2026-10-30 deadline snapshot')
    args = parser.parse_args()
    result = analyze(safe_path(args.root))
    safe_path(args.out).write_text(json.dumps(result, indent=1)+'\n', encoding='utf-8')
    if args.report:
        assert args.final_sha
        safe_path(args.report).write_text(report(result, args.final_sha, args.stamp), encoding='utf-8')
    print({arm: (s['n_starts'], s['n_worlds'], s['J300']['paired_delta'])
           for arm, s in result['per_arm'].items()})


if __name__ == '__main__':
    main()
