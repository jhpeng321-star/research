"""Local (Windows) CPU tests only: provide flock when fcntl is absent; no-op on Linux."""
from tools.coopmwm.fcntl_compat import install

install()
