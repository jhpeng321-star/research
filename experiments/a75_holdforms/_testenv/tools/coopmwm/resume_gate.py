"""MAIN02:24 owner-approved cap; every earlier resume cap is superseded."""
RESUME_CAP=7500000


def check_budget(plan):
    if plan['allocated']>RESUME_CAP:
        raise ValueError(f"owner-approved cap {RESUME_CAP}: registered allocation {plan['allocated']} exceeds by "
            f"{plan['allocated']-RESUME_CAP}; resolve owner cap/scope before new SUMO dispatch")
