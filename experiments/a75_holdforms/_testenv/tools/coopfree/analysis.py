"""Reuse peer detectors verbatim; cluster continuous learned actions independently."""
import gzip
import json
import numpy as np
from .protocol import checked,sha


def tape(path,seed):
    from tools.coopbehav.io import Tape,Vehicle
    with gzip.open(checked(path),'rt',encoding='utf-8') as f:
        frames = [json.loads(s) for s in f]
    rows = []
    for frame in frames:
        rows.append({r['vid']:Vehicle(r['s_m'],r['v_mps'],r['length_m'],r['lane'],r['is_cav'],
                     r.get('issued_acceleration_mps2'),r.get('controller'),
                     r.get('policy_mean_acceleration_mps2')) for r in frame['vehicles']})
    return Tape(np.asarray([f['time_s'] for f in frames]),rows,meta=dict(seed=seed),
                coverage=dict(trace_scope='full_observable_fleet'))


def analyse(root,groups):
    from tools.coopbehav.detectors import detect
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    events,x = [],[]
    for g in groups:
        actual_path = root/'jobs'/g['joint_id']/'observable.jsonl.gz'
        baseline_path = root/'jobs'/g['baseline_id']/'observable.jsonl.gz'
        actual,baseline = tape(actual_path,g['world']['seed']),tape(baseline_path,g['world']['seed'])
        found = detect(actual,baseline,g['sample']['context']['members'],reference_kind='policy_mean_reference')
        events.append(dict(joint_id=g['joint_id'],events=found['events'],coverage=found['coverage'],trace_sha256=sha(actual_path),
                           reference_sha256=sha(baseline_path)))
        a = np.asarray(g['sample']['action'])-np.asarray(g['sample']['context']['anchors'])
        x.append([*a.mean(0),*a.std(0),g['sample']['window'],len(a)])
    clusters = dict(status='insufficient distinct groups',labels=[],k=None)
    if len(x) >= 3 and len(np.unique(np.asarray(x),axis=0)) >= 3:
        z = StandardScaler().fit_transform(x)
        trials = []
        for k in range(2,min(8,len(x)-1,len(np.unique(z,axis=0)))+1):
            model = KMeans(n_clusters=k,n_init=10,random_state=20261101).fit(z)
            if len(np.unique(model.labels_)) > 1:
                trials.append((float(silhouette_score(z,model.labels_)),k,model.labels_.tolist()))
        if trials:
            score,k,labels = max(trials,key=lambda t:(t[0],-t[1]))
            clusters = dict(status='descriptive',k=k,silhouette=score,labels=labels,
                            features=['speed_residual_mean','loggap_residual_mean','speed_residual_sd',
                                      'loggap_residual_sd','window_s','group_size'])
    return dict(detectors='S-COOPBEHAV existing sign/geometry detectors, source hashes in binding',
                events=events,clusters=clusters,
                note='Detector candidates and clusters are not causal mechanism claims; inspect paired support and C_G.')
