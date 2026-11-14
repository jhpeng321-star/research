from counterfactual.actionability import dose_qualified, summarize_actionability


def _row(seed: int, step: int, *, safe: bool, selected: bool) -> dict:
    return {
        "case_id": f"case-{seed}-{step}",
        "seed": seed,
        "decision_step": step,
        "features": {"signal": float(seed), "constant": 1.0},
        "treatment_dose": {
            "activation_feasible": True,
            "realized_duration_fraction": 1.0,
            "lane_coverage_fraction": 1.0,
            "assembly_completed": True,
            "holding_fraction_of_override": 0.75,
            "wall_geometry": {
                "geometry_mode": "compact_moving",
                "fraction_within_configured_span": 1.0,
            },
        },
        "safety_service_oracle": {
            "safety_admissible": safe,
            "service_noninferior": selected,
            "wall_selected": selected,
            "refusal_reasons": [] if selected else ["safety_budget"],
            "effects": {},
        },
        "arms": {
            "1": {
                "outcomes": {
                    "hard_brake_attribution": {"event_starts": []},
                    "low_ttc_attribution": {"events": []},
                }
            }
        },
    }


def test_actionability_reports_step_and_feature_heterogeneity() -> None:
    rows = [
        _row(1, 120, safe=False, selected=False),
        _row(2, 120, safe=False, selected=False),
        _row(3, 180, safe=True, selected=True),
        _row(4, 180, safe=True, selected=True),
    ]
    assert all(dose_qualified(row) for row in rows)

    report = summarize_actionability(rows)

    assert report["dose_qualified_case_count"] == 4
    assert report["safe_case_count"] == 2
    assert report["selected_seed_count"] == 2
    assert report["by_decision_step"]["180"]["selected_count"] == 2
    top = report["predecision_separation"]["safe_vs_unsafe_completed"]["top"]
    assert top[0]["feature"] == "signal"


def test_actionability_accepts_current_traffic_noninferior_oracle_key() -> None:
    row = _row(1, 120, safe=True, selected=True)
    oracle = row["safety_service_oracle"]
    oracle["traffic_noninferior"] = oracle.pop("service_noninferior")
    report = summarize_actionability([row])
    assert report["service_noninferior_case_count"] == 1
