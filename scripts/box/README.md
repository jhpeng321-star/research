# Box launcher

`launch.sh preflight|run [section.key=value ...]` exports the box data locations as `WMS_DATA_<NAME>`
and calls `python -m wms train --config "$CONFIG" --mode MODE`. It needs `SW_ROOT` (switching export);
`CONFIG` defaults to `configs/r3.toml`, `W` sets a fresh run directory, `PY` the interpreter.
The file must keep LF line endings (`.gitattributes` enforces it; the script refuses CRLF).
