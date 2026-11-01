"""Explicit fresh-session cap overrides a historical dispatcher's larger quote."""
RESUME_CAP=6000000


def check_budget(plan):
    if plan['allocated']>RESUME_CAP:
        raise ValueError(f"fresh resume cap {RESUME_CAP}: registered allocation {plan['allocated']} exceeds by "
            f"{plan['allocated']-RESUME_CAP}; resolve owner cap/scope before new SUMO dispatch")
