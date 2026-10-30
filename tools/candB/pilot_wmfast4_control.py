"""Exclusive S4 waiter, after cooperative retirement of the prior S2 waiter."""
import inspect

from tools.candB import pilot_wmfast_control as original
from tools.candB.pilot_continue import replace_once


def controller_main():
    source = inspect.getsource(original.main)
    source = source.replace('WMFAST_CONTROL', 'WMFAST4_CONTROL')
    source = replace_once(source, 'pilot_WMFAST_CPU_RESULT.json', 'pilot_WMFAST4_CPU_RESULT.json')
    source = replace_once(source, 'tools.candB.pilot_wmfast_launch', 'tools.candB.pilot_wmfast4_launch')
    source = replace_once(source, "observation['free_MiB'] >= 38000", "observation['free_MiB'] >= 46000")
    namespace = dict(original.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace['main']


if __name__ == '__main__':
    controller_main()()
