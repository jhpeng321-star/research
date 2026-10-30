"""Run one frozen verdict producer under the same restricted-input guard."""
import runpy
import sys

from tools.wmfixmerge.data import install_guard, safe


if __name__ == "__main__":
    script = str(safe(sys.argv[1]))
    sys.argv = sys.argv[1:]
    install_guard()
    runpy.run_path(script, run_name="__main__")
