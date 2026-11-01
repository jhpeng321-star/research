"""MAIN01:53 confirms the owner-approved cap; earlier resume caps are superseded."""
RESUME_CAP=6500000


def check_budget(plan):
    if plan['allocated']>RESUME_CAP:
        raise ValueError(f"owner-approved cap {RESUME_CAP}: registered allocation {plan['allocated']} exceeds by "
            f"{plan['allocated']-RESUME_CAP}; resolve owner cap/scope before new SUMO dispatch")
