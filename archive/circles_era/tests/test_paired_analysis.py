from output.collapse_prevention_audit_v1.paired_analysis import classify_episodes, outcome_for_pair, active_at_entry, confirmed_primary

def test_straddling_is_separate_from_preexisting():
    x = classify_episodes([{"candidate_onset_s": 590, "confirmation_s": 649}])
    assert len(x["straddling"]) == 1 and not x["preexisting"]

def test_absence_is_window_bounded():
    b = {"episodes": [{"kind":"extended", "candidate_onset_s": 700, "confirmation_s": 759}]}
    t = {"episodes": []}
    assert outcome_for_pair(b, t)["category"] == "absent_within_window"

def test_earlier_and_delayed_are_distinct():
    b = {"episodes": [{"kind":"extended", "candidate_onset_s": 700, "confirmation_s": 759}]}
    assert outcome_for_pair(b, {"episodes": [{"kind":"extended", "candidate_onset_s": 650, "confirmation_s": 709}]})["category"] == "earlier"
    assert outcome_for_pair(b, {"episodes": [{"kind":"extended", "candidate_onset_s": 800, "confirmation_s": 859}]})["category"] == "delayed"

def test_preexisting_recovered_before_entry_is_clean():
    b={"episodes":[{"kind":"extended","candidate_onset_s":400,"confirmation_s":459,"recovery_confirmation_s":550},{"kind":"extended","candidate_onset_s":700,"confirmation_s":759}]}
    assert outcome_for_pair(b,{"episodes":[]})["category"] == "absent_within_window"

def test_preexisting_still_active_is_not_clean():
    b={"episodes":[{"kind":"extended","candidate_onset_s":400,"confirmation_s":459}]}
    assert active_at_entry(b["episodes"])

def test_novel_induced_event_and_unconfirmed_tail():
    b={"episodes":[]}
    t={"episodes":[{"kind":"extended","candidate_onset_s":700,"confirmation_s":759},{"kind":"extended","candidate_onset_s":1141,"confirmation_s":None}]}
    assert outcome_for_pair(b,t)["category"] == "treatment_induced"
    assert len(confirmed_primary(t["episodes"])) == 1
