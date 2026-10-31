"""Match the registered Python3.10 scalar metric reducer on Python3.12 hosts.

Python3.12 changed built-in float sum to compensated summation. This shim is scoped
to metric/observable-summary modules; it never patches the policy or safety layer.
"""


def reference_sum(values, start=0):
    total=start
    for value in values:
        total=total+value
    return total


def install():
    import traffic_shepherd.c4.env as env
    import tools.coopbandit.protocol as protocol
    import tools.leverprobe.selection as selection
    from wms.control import safety
    original_safety=safety.safe_accel
    env.sum=reference_sum
    protocol.sum=reference_sum
    selection.sum=reference_sum
    assert safety.safe_accel is original_safety
