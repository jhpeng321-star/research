"""Deterministic, CPU-only exploratory clustering of positive measured gains.

``discover(records)`` accepts mappings with ``id``, ``C_G``, ``features`` (a
mapping of physical feature names to numbers), and ``explained_by`` (detector
names). Only finite, strictly positive numeric C_G values enter the ranking;
missing gains are never inferred from features or descriptive arm labels.

Feature means and population standard deviations (ddof=0) are fitted to the
finite observations of ranked records. Missing/non-numeric/non-finite features
are mean-imputed (standardized value zero). Records with no finite feature are
ranked but unassigned. Columns whose std/max(abs(observation)) <= float64 eps
are constant at numerical resolution and excluded from distance calculations.
Scaling is computed in normalized units to avoid overflow on finite inputs.

Clustering is Euclidean average-linkage agglomeration. For N clusterable rows,
choose the largest mean silhouette among k=2..floor(sqrt(N)), breaking numerical
ties toward smaller k. Singletons have silhouette zero. Scores must exceed
N*float64_eps; otherwise use k=1, including constant/no-distance data and N<4.
Linkage ties use row-major order after sorting by descending C_G and canonical
JSON id. Cluster labels follow their first ranked member. No RNG is used.
This exact, exploratory algorithm takes O(N**2) memory and O(N**3) time.

The result is JSON-serializable (including with allow_nan=False). ``assignments``
contains every ranked record, with a null cluster_id for unassignable rows.
``features`` describes scaling, ``clusters`` describes physical means and
detector coverage, and ``unanticipated_candidates`` lists wholly unexplained
clusters. These candidates are associations for investigation, not causal proof.
Statuses are ok, degenerate (one cluster), no_usable_features, or no_positive_c_g.
IDs must be unique JSON-serializable values among ranked records.
"""

from collections.abc import Mapping
import json
import math
from numbers import Real

import numpy as np


def _finite_number(value):
    """Do not coerce strings, booleans, or missing values into measurements."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _mean(values):
    magnitude = float(np.max(np.abs(values)))
    return float(np.mean(values / magnitude) * magnitude) if magnitude else 0.0


def _silhouette(distances, groups):
    scores = np.zeros(len(distances), dtype=float)
    for group in groups:
        if len(group) == 1:
            continue
        within = distances[np.ix_(group, group)].sum(axis=1) / (len(group) - 1)
        between = np.min(
            [distances[np.ix_(group, other)].mean(axis=1)
             for other in groups if other is not group], axis=0
        )
        denominator = np.maximum(within, between)
        scores[group] = np.divide(
            between - within, denominator,
            out=np.zeros_like(within), where=denominator > 0,
        )
    return float(scores.mean())


def _partition(points):
    n = len(points)
    k_max = math.isqrt(n)
    tolerance = n * np.finfo(float).eps
    method = {
        "name": "standardized_euclidean_average_linkage_silhouette",
        "scaling": "finite-observation mean and population std (ddof=0); mean imputation",
        "constant_rule": "std / max_abs <= float64 epsilon (or all zero)",
        "k_candidates": list(range(2, k_max + 1)),
        "selection": "maximum silhouette > N*eps; ties within N*eps prefer smaller k; otherwise k=1",
        "singleton_silhouette": 0.0,
        "tie_break": "descending C_G, canonical JSON id, row-major linkage ties",
        "silhouette_scores": [],
        "selected_k": 1 if n else 0,
    }
    fallback = [list(range(n))] if n else []
    if n < 4 or points.shape[1] == 0:
        return fallback, method
    distances = np.zeros((n, n), dtype=float)
    for column in points.T:
        distances += (column[:, None] - column[None, :]) ** 2
    np.sqrt(distances, out=distances)
    if not np.any(distances):
        return fallback, method

    linkage = distances.copy()
    np.fill_diagonal(linkage, np.inf)
    groups = {i: [i] for i in range(n)}
    partitions = {}
    while len(groups) >= 2:
        if len(groups) <= k_max:
            partitions[len(groups)] = [members[:] for members in groups.values()]
        if len(groups) == 2:
            break
        left, right = np.unravel_index(np.argmin(linkage), linkage.shape)
        left_size, right_size = len(groups[left]), len(groups[right])
        for other in groups:
            if other not in (left, right):
                distance = (
                    left_size * linkage[left, other]
                    + right_size * linkage[right, other]
                ) / (left_size + right_size)
                linkage[left, other] = linkage[other, left] = distance
        groups[left] = sorted(groups[left] + groups.pop(right))
        linkage[right, :] = linkage[:, right] = np.inf
        linkage[left, left] = np.inf

    best, best_score = fallback, tolerance
    for k in sorted(partitions):
        score = _silhouette(distances, partitions[k])
        method["silhouette_scores"].append({"k": k, "score": score})
        if (len(best) == 1 and score > tolerance) or score > best_score + tolerance:
            best, best_score = partitions[k], score
    best.sort(key=min)
    method["selected_k"] = len(best)
    return best, method


def discover(records):
    """Return ranked positive-gain records and exploratory physical clusters.

    Input mappings are never modified. Feature names must be strings, and
    explained_by must be a list of detector-name strings when supplied.
    Invalid measurements are treated as missing, rather than as zero.
    """
    ranked = []
    input_count = 0
    seen_ids = set()
    for record in records:
        input_count += 1
        gain = _finite_number(record.get("C_G"))
        if gain is None or gain <= 0:
            continue
        identifier = record["id"]
        id_key = json.dumps(identifier, sort_keys=True, allow_nan=False)
        if id_key in seen_ids:
            raise ValueError("Ranked record IDs must be unique")
        seen_ids.add(id_key)
        features = record.get("features")
        features = features if isinstance(features, Mapping) else {}
        if any(not isinstance(name, str) for name in features):
            raise ValueError("Physical feature names must be strings")
        explained_by = record.get("explained_by", [])
        if not isinstance(explained_by, list) or any(
            not isinstance(name, str) for name in explained_by
        ):
            raise ValueError("explained_by must be a list of detector names")
        ranked.append({
            "id": json.loads(id_key), "id_key": id_key, "C_G": gain,
            "features": {name: _finite_number(value) for name, value in features.items()},
            "explained_by": explained_by[:],
        })
    ranked.sort(key=lambda row: (-row["C_G"], row["id_key"]))
    assignments = [
        {"id": row["id"], "rank": rank, "C_G": row["C_G"], "cluster_id": None}
        for rank, row in enumerate(ranked, 1)
    ]
    eligible = []
    for index, row in enumerate(ranked):
        if any(value is not None for value in row["features"].values()):
            eligible.append(index)
        else:
            assignments[index]["reason"] = "no_finite_physical_features"
    names = sorted({name for row in ranked for name in row["features"]})
    rows = [ranked[index] for index in eligible]
    raw = np.array([
        [row["features"].get(name) for name in names] for row in rows
    ], dtype=float).reshape(len(rows), len(names))
    standardized = np.zeros_like(raw)
    feature_info = []
    used_columns = []
    for column, name in enumerate(names):
        observed = np.isfinite(raw[:, column])
        values = raw[observed, column]
        magnitude = float(np.max(np.abs(values))) if len(values) else 0.0
        normalized = values / magnitude if magnitude else np.zeros_like(values)
        center = float(normalized.mean()) if len(values) else 0.0
        deviation = float(np.sqrt(np.mean((normalized - center) ** 2))) if len(values) else 0.0
        used = deviation > np.finfo(float).eps
        if used:
            standardized[observed, column] = (normalized - center) / deviation
            used_columns.append(column)
        feature_info.append({
            "name": name,
            "mean": float(center * magnitude) if len(values) else None,
            "scale": float(deviation * magnitude) if len(values) else None,
            "observed_count": int(observed.sum()),
            "imputed_count": int(len(rows) - observed.sum()),
            "used_for_clustering": bool(used),
        })
    groups, method = _partition(standardized[:, used_columns])
    clusters = []
    candidates = []
    for number, group in enumerate(groups, 1):
        cluster_id = "cluster_{}".format(number)
        members = [rows[index] for index in group]
        physical_means = {}
        signed_means = {}
        descriptions = []
        for column, name in enumerate(names):
            values = raw[group, column]
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            physical_means[name] = _mean(values)
            signed = float(standardized[group, column].mean())
            signed_means[name] = signed
            label = name.replace("_", " ").replace("-", " ")
            direction = "Higher" if signed > len(rows) * np.finfo(float).eps else (
                "Lower" if signed < -len(rows) * np.finfo(float).eps else "Baseline"
            )
            descriptions.append(
                "{} mean {} (physical mean {:+.6g}; {:+.3g} SD relative to ranked-record mean)".format(
                    direction, label, physical_means[name], signed,
                )
            )
        explanations = sorted({name for row in members for name in row["explained_by"]})
        unexplained = not any(row["explained_by"] for row in members)
        cluster = {
            "cluster_id": cluster_id,
            "member_ids": [row["id"] for row in members],
            "size": len(members),
            "mean_C_G": _mean(np.array([row["C_G"] for row in members])),
            "physical_means": physical_means,
            "standardized_means": signed_means,
            "description": "; ".join(descriptions) + ".",
            "explained_by": explanations,
            "unexplained": unexplained,
        }
        clusters.append(cluster)
        for index in group:
            assignments[eligible[index]]["cluster_id"] = cluster_id
        if unexplained:
            candidates.append({
                "cluster_id": cluster_id,
                "member_ids": cluster["member_ids"][:],
                "description": cluster["description"],
                "interpretation": "Unanticipated candidate: no member has a detector explanation; not causal proof.",
            })
    status = "ok" if len(groups) > 1 else "degenerate"
    if not ranked:
        status = "no_positive_c_g"
    elif not rows:
        status = "no_usable_features"
    return {
        "status": status, "method": method, "features": feature_info,
        "clusters": clusters, "assignments": assignments,
        "unanticipated_candidates": candidates,
        "counts": {"input": input_count, "ranked": len(ranked), "clustered": len(rows)},
    }
