"""Narrow source overlay: configurable canary boxes, preserving the historical default."""


def single_canary_source(source):
    before = "for box_name in ('cpuP', 'cpuQ'):"
    after = "for box_name in canary.get('boxes', ('cpuP', 'cpuQ')):"
    assert source.count(before) == 1 and after not in source
    return source.replace(before, after)
