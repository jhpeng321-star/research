"""Narrow resume repair; completed bootstrap batches have no new startup canary."""


def resume_source(source):
    old = "if canary.get('cycle') == k and batch_index == 0:"
    new = "if canary.get('cycle') == k and batch_index == 0 and ordered:"
    assert source.count(old) == 1 and new not in source
    return source.replace(old, new)
