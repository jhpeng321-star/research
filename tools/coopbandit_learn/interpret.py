"""Feature importance and partial dependence for the fitted per-option GBTs.

Plots use the matplotlib Agg backend (headless, CPU only). The fitted models
are the full-data per-option models from a CVResult; importance comes from the
GBT impurity importances, partial dependence from sklearn's exact enumeration
for the small depth-2 ensembles.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cv import CVResult
from .schema import OPTIONS, checked


def importance_table(result: CVResult):
    rows = []
    for option, model in result.models_full.items():
        if model is None:
            continue
        for name, value in zip(result.feature_names, model.feature_importances_):
            rows.append(dict(option=option, feature=name, importance=float(value)))
    return rows


def partial_dependence_table(result: CVResult, X, top_k=4, grid=9):
    rows = []
    X = np.asarray(X, dtype=float)
    for option, model in result.models_full.items():
        if model is None:
            continue
        order = np.argsort(model.feature_importances_)[::-1][:top_k]
        for j in order:
            name = result.feature_names[int(j)]
            values = np.unique(np.quantile(X[:, j], np.linspace(0.0, 1.0, grid)))
            for v in values:
                modified = X.copy()
                modified[:, j] = v
                rows.append(dict(option=option, feature=name, value=float(v),
                                 mean_predicted_D=float(model.predict(modified).mean())))
    return rows


def _plot_option(ax, rows, option):
    sub = [r for r in rows if r["option"] == option]
    for feature in dict.fromkeys(r["feature"] for r in sub):
        pts = sorted((r["value"], r["mean_predicted_D"]) for r in sub if r["feature"] == feature)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=feature)
    ax.set_title(option)
    ax.set_xlabel("feature value")
    ax.set_ylabel("predicted D (veh-s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)


def plots(result: CVResult, importance_rows, pdp_rows, out_dir):
    out_dir = checked(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = [o for o in OPTIONS if result.models_full.get(o) is not None]
    fig, axes = plt.subplots(len(options), 1, figsize=(7, 3.0 * len(options)), squeeze=False)
    for ax, option in zip(axes[:, 0], options):
        _plot_option(ax, pdp_rows, option)
    fig.suptitle("Partial dependence: predicted D600 vs top features (full-data GBT)")
    fig.tight_layout()
    fig.savefig(out_dir / "partial_dependence.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(options), figsize=(3.2 * len(options), 4.2), squeeze=False)
    for ax, option in zip(axes[0], options):
        sub = sorted((r for r in importance_rows if r["option"] == option),
                     key=lambda r: -r["importance"])[:10]
        y = np.arange(len(sub))
        ax.barh(y, [r["importance"] for r in sub])
        ax.set_yticks(y, [r["feature"] for r in sub], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(option, fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("GBT feature importance (top 10 per option)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    return [out_dir / "partial_dependence.png", out_dir / "feature_importance.png"]


def write_tables(out_dir, importance_rows, pdp_rows):
    out_dir = checked(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("feature_importance.csv", importance_rows),
                       ("partial_dependence.csv", pdp_rows)):
        with open(checked(out_dir / name), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
