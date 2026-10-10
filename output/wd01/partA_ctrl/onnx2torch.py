"""Minimal ONNX-protobuf reader + a faithful torch graph executor.

Part A step A2 (WD-01): the frozen CIRCLES model (baselines/circles/
model.onnx, sha256 pinned in the adopted adapter config) is executed as a
torch nn.Module so the damping commanded-speed path becomes batched and
device-agnostic with ZERO simulation.  The reader implements just enough
of the protobuf wire format for THIS graph's op set (Constant, Div, Gemm,
Tanh, Clip, ReduceMax, Mul, Add, Sub, Exp); weights are extracted from the
original file and re-registered as float32 buffers.

No onnx/onnxruntime dependency here: only torch + stdlib.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

_ONNX_DTYPES = {1: ("float32", 4), 7: ("int64", 8), 6: ("int32", 4)}


def _read_varint(buf: bytes, i: int):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _iter_fields(buf: bytes):
    """Yield (field_number, wire_type, value) over one protobuf message."""
    i = 0
    end = len(buf)
    while i < end:
        key, i = _read_varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            value, i = _read_varint(buf, i)
        elif wt == 2:
            length, i = _read_varint(buf, i)
            value = buf[i:i + length]
            i += length
        elif wt == 5:
            value = buf[i:i + 4]
            i += 4
        elif wt == 1:
            value = buf[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fn, wt, value


def _group(buf: bytes) -> dict:
    """field_number -> list of (wire_type, value), order preserved."""
    out: dict[int, list] = {}
    for fn, wt, value in _iter_fields(buf):
        out.setdefault(fn, []).append((wt, value))
    return out


def _parse_tensor(buf: bytes) -> np.ndarray:
    fields = _group(buf)
    dims = [v for _, v in fields.get(1, [])]
    dtype_code = fields.get(2, [(0, 1)])[0][1]
    name_end = fields.get(8)
    name = name_end[0][1].decode("utf-8") if name_end else ""
    if dtype_code != 1:
        raise NotImplementedError(
            f"tensor {name!r}: only float32 initializers are supported "
            f"(got dtype {dtype_code})")
    if 9 in fields:  # raw_data
        raw = fields[9][0][1]
        array = np.frombuffer(raw, dtype="<f4").copy()
    elif 4 in fields:  # packed float_data
        raw = fields[4][0][1]
        array = np.frombuffer(raw, dtype="<f4").copy()
    else:  # unpacked float_data (fixed32)
        array = np.array([struct.unpack("<f", v)[0]
                          for _, v in fields.get(4, [])], dtype=np.float32)
    return array.reshape(dims) if dims else array.reshape(1) if array.size else array


def _parse_attribute(buf: bytes):
    fields = _group(buf)
    name = fields.get(1, [(2, b"?")])[0][1].decode("utf-8")
    type_code = fields.get(20, [(0, 0)])[0][1]

    def ints(fns):
        values: list[int] = []
        for wt, v in fields.get(fns, []):
            if wt == 0:
                values.append(v)
            else:
                values.extend(_read_packed_ints(v))
        return values

    if type_code == 2:  # INT
        return name, ints(3)[0] if ints(3) else 0
    if type_code == 1:  # FLOAT
        wt, v = fields.get(2, [(5, b"\0\0\0\0")])[0]
        return name, struct.unpack("<f", v)[0]
    if type_code == 3:  # STRING
        return name, fields.get(4, [(2, b"")])[0][1].decode("utf-8")
    if type_code == 4:  # TENSOR
        return name, _parse_tensor(fields.get(5)[0][1])
    if type_code == 7:  # INTS
        return name, ints(8)
    if type_code == 6:  # FLOATS
        values: list[float] = []
        for wt, v in fields.get(7, []):
            if wt == 2:
                values.extend(struct.unpack(f"<{len(v)//4}f", v))
            else:
                values.append(struct.unpack("<f", v)[0])
        return name, values
    raise NotImplementedError(f"attribute {name!r}: type {type_code}")


def _read_packed_ints(buf: bytes) -> list[int]:
    values, i = [], 0
    while i < len(buf):
        v, i = _read_varint(buf, i)
        values.append(v)
    return values


class OnnxGraph:
    """A parsed ONNX graph: initializers + ordered nodes + io names."""

    def __init__(self, model_path: str | Path):
        data = Path(model_path).read_bytes()
        top = _group(data)
        graph_buf = top.get(7)
        if graph_buf is None:
            raise ValueError("not an ONNX ModelProto (no graph field)")
        graph = _group(graph_buf[0][1])
        self.initializers: dict[str, np.ndarray] = {}
        for wt, tensor_buf in graph.get(5, []):
            t = _group(tensor_buf)
            tname = t.get(8, [(2, b"")])[0][1].decode("utf-8")
            self.initializers[tname] = _parse_tensor(tensor_buf)
        self.nodes: list[dict] = []
        for _, node_buf in graph.get(1, []):
            fields = _group(node_buf)
            node = {
                "op": fields.get(4, [(2, b"")])[0][1].decode("utf-8"),
                "name": (fields.get(3, [(2, b"")])[0][1].decode("utf-8")
                         if 3 in fields else ""),
                "inputs": [v.decode("utf-8") for _, v in fields.get(1, [])],
                "outputs": [v.decode("utf-8") for _, v in fields.get(2, [])],
                "attrs": dict(_parse_attribute(b) for _, b in fields.get(5, [])),
            }
            self.nodes.append(node)
        self.input_names = [
            _group(v[1]).get(1, [(2, b"")])[0][1].decode("utf-8")
            for v in graph.get(11, [])]
        self.output_names = [
            _group(v[1]).get(1, [(2, b"")])[0][1].decode("utf-8")
            for v in graph.get(12, [])]
        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise ValueError("expected a single-input single-output graph")

    @classmethod
    def from_bytes(cls, data: bytes) -> "OnnxGraph":
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
            fh.write(data)
            path = fh.name
        try:
            return cls(path)
        finally:
            Path(path).unlink(missing_ok=True)


class TorchOnnxGraph(torch.nn.Module):
    """Execute a parsed ONNX graph in torch, weights as registered buffers.

    Mirrors the reference graph op-for-op in float32 (ORT default); the
    module is device-agnostic (`.to(device)`) and batched over rows.

    The frozen CIRCLES graph is a SINGLETON graph (the reference calls it
    with one [1, 3] row only): its rank-1 intermediates accidentally
    broadcast against the [N, 1] Gemm outputs for N > 1 (ORT on [N, 3]
    yields an [N, N] outer product, not N accelerations).  ``forward``
    therefore implements vectorized-singleton semantics: every
    intermediate that would be a singleton's trailing [1] axis is kept as
    a per-row scalar [N].  Per-row equivalence with onnxruntime singleton
    runs is a pinned test (E1).
    """

    def __init__(self, graph: OnnxGraph):
        super().__init__()
        self._graph = graph
        for name, array in graph.initializers.items():
            safe = "w_" + name.replace(".", "_")
            self.register_buffer(safe, torch.from_numpy(array.copy()))
        self._weights = {name: safe
                         for name, safe in zip(graph.initializers,
                                               ["w_" + n.replace(".", "_")
                                                for n in graph.initializers])}
        self.input_name = graph.input_names[0]
        self.output_name = graph.output_names[0]

    def _resolve(self, name: str, env: dict) -> torch.Tensor:
        if name in env:
            return env[name]
        if name in self._graph.initializers:
            return getattr(self, self._weights[name])
        if name == "":  # optional empty input
            return None
        raise KeyError(f"unknown graph value {name!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        env: dict[str, torch.Tensor] = {self.input_name: x}

        def put(name: str, value: torch.Tensor) -> None:
            # vectorized-singleton discipline: a singleton's [1, 1] Gemm
            # output is one scalar per row -> [N]
            if value.dim() == 2 and value.shape == (batch, 1):
                value = value.squeeze(1)
            env[name] = value

        for node in self._graph.nodes:
            op = node["op"]
            ins = [self._resolve(n, env) for n in node["inputs"]]
            attrs = node["attrs"]
            out = self._run_op(op, ins, attrs)
            put(node["outputs"][0], out)
        out = env[self.output_name]
        if out.dim() == 2 and out.shape == (batch, 1):
            out = out.squeeze(1)
        return out

    @staticmethod
    def _run_op(op: str, ins, attrs: dict) -> torch.Tensor:
        if op == "Constant":
            value = attrs.get("value")
            if not isinstance(value, np.ndarray):
                raise NotImplementedError("Constant without a tensor value")
            return torch.from_numpy(value.copy())
        if op == "Div":
            return ins[0] / ins[1]
        if op == "Mul":
            return ins[0] * ins[1]
        if op == "Add":
            return ins[0] + ins[1]
        if op == "Sub":
            return ins[0] - ins[1]
        if op == "Exp":
            return torch.exp(ins[0])
        if op == "Tanh":
            return torch.tanh(ins[0])
        if op == "Gemm":
            a, b = ins[0], ins[1]
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            if attrs.get("transA", 0):
                a = a.transpose(-1, -2)
            if attrs.get("transB", 0):
                b = b.transpose(-1, -2)
            y = alpha * torch.matmul(a, b)
            if len(ins) > 2 and ins[2] is not None:
                y = y + beta * ins[2]
            return y
        if op == "Clip":
            x = ins[0]
            lo = attrs.get("min")
            hi = attrs.get("max")
            return torch.clamp(x, None if lo is None else float(lo),
                               None if hi is None else float(hi))
        if op in ("ReduceMax", "ReduceMin"):
            axes = attrs.get("axes")
            keepdim = bool(attrs.get("keepdims", 1))
            dims = tuple(int(a) for a in axes) if axes not in (None, []) else None
            fn = torch.amax if op == "ReduceMax" else torch.amin
            return fn(ins[0], dim=dims, keepdim=keepdim)
        raise NotImplementedError(f"op {op} not supported by this mini-executor")
