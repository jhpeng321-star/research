"""Training-data layer: pack corpus reader, window selection, datasets, sidecars.

Dependency rule (docs/ARCHITECTURE_V4.md 2.1): imports ``wms.core`` and the vendored
legacy pack reader only; never model, train or eval code.  ``registry.py`` (path
resolution) is owned by the config/registry package and is not imported here.
"""
