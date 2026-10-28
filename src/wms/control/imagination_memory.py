"""Exact-objective memory execution for Dreamer (all switches default off).

The outer graph sees the ORIGINAL full batch. Its VJP is replayed in start
microbatches, including every member/future/variant of those starts. This avoids
approximating quantiles, masked reductions, uncertainty, or cooperation weights.
No optimizer, target, temperature, or return-statistic update happens here.
"""
from __future__ import annotations

import copy
import inspect
import json
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.utils.checkpoint import checkpoint

from .counterfactual import Variants, expand_rows


def _trace_memory(trainer, event, **details):
    """Owned execution diagnostics; no synchronization, RNG or objective changes."""
    if trainer.output_dir is None:
        return
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    record = {'t': time.time(), 'event': event, 'update': trainer.update_count, **details}
    if trainer.device.type == 'cuda':
        record.update(allocated_bytes=torch.cuda.memory_allocated(trainer.device),
                      reserved_bytes=torch.cuda.memory_reserved(trainer.device))
    with (trainer.output_dir / 'memory_execution.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


@contextmanager
def _saved_tensor_spill(trainer):
    """Spill activations, sharing repeated immutable parameter-storage copies.

    save_on_cpu copies the same WM/actor matrix on every use. At K8/90s this
    can exhaust host RAM despite segment checkpointing. Persistent storages
    already owned by the trainer can safely share a CPU snapshot per version
    and view geometry; transient activations retain the original copy semantics.
    """
    persistent = {}
    modules = [*trainer.world_models, trainer.policy, trainer.group_encoder, trainer.critic]
    for module in modules:
        for tensor in (*module.parameters(), *module.buffers()):
            storage = tensor.untyped_storage()
            persistent[(tensor.device, storage.data_ptr())] = storage
    copies = {}
    pin = trainer.device.type == 'cuda'
    # S-POLSPEED ``WMS_SPEED_ASYNC_SPILL``: enqueue the device->pinned-host copy on the
    # current stream instead of blocking the host once per saved tensor. Same bytes: the
    # caching host allocator records the copy's stream event before a pinned block is
    # reused, the source block cannot be reused by later kernels before the stream-ordered
    # copy has read it, and the host buffer is only ever read back by a stream-ordered
    # host->device copy in ``unpack``. Default OFF.
    from wms.core import speed_flags
    async_copy = pin and speed_flags.enabled("async_spill")
    stats = {'copied_bytes': 0, 'copies': 0, 'reuses': 0}
    trainer.memory_spill_stats = stats

    def pack(tensor):
        if tensor.device.type == 'cpu':
            return tensor.device, tensor
        storage = tensor.untyped_storage()
        key = None
        if (tensor.device, storage.data_ptr()) in persistent:
            key = (tensor.device, storage.data_ptr(), tensor._version, tensor.storage_offset(),
                   tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
            if key in copies:
                stats['reuses'] += 1
                return tensor.device, copies[key]
        # Match save_on_cpu's contiguous host layout as well as its values;
        # preserving a transposed weight stride can select a different GEMM.
        cpu = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout,
                          device='cpu', pin_memory=pin)
        cpu.copy_(tensor.detach(), non_blocking=async_copy)
        stats['copies'] += 1
        stats['copied_bytes'] += tensor.numel() * tensor.element_size()
        if key is not None:
            copies[key] = cpu
        return tensor.device, cpu

    def unpack(packed):
        device, cpu = packed
        return cpu.to(device, non_blocking=pin)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def _select(value, indices, total):
    return expand_rows(value, indices, total)


def _current_tape():
    return next((mode for mode in torch.utils._python_dispatch._get_current_dispatch_mode_stack()
                 if isinstance(mode, _RandomTape)), None)


def capture_future(cache, collections):
    """Keep detached forward values and exact stochastic trace boundary."""
    tape = _current_tape()
    payload = {name: tree_map(lambda x: x.detach() if torch.is_tensor(x) else x, values[-1])
               for name, values in collections.items() if values}
    cache['items'].append({'payload': payload, 'tape_end': tape.position})


def replay_cached_future(cache, future, collections, rows, total_rows):
    """Inactive futures remain constants; outer full-K VJP still sees all values."""
    saved = cache['items'][future]
    for name, value in saved['payload'].items():
        axis = 0 if name == 'states' else 1
        def select(tensor):
            if torch.is_tensor(tensor) and tensor.ndim > axis and tensor.shape[axis] == total_rows:
                return tensor.index_select(axis, rows.to(tensor.device))
            return tensor
        collections[name].append(tree_map(select, value))
    _current_tape().position = saved['tape_end']


def _subset(batch, variants, start, stop):
    total = batch.state["s_m"].shape[0]
    indices = torch.arange(start, stop, device=batch.state["s_m"].device)
    spec = batch.entry_spec
    if spec is not None:
        spec = {k: v[start:stop] if k in ("histories", "seeds", "queues") else v
                for k, v in spec.items()}
    small = replace(batch, state=_select(batch.state, indices, total),
                    rollout_kwargs=_select(batch.rollout_kwargs, indices, total),
                    metadata=batch.metadata[start:stop], entry_spec=spec,
                    entry_feat=_select(batch.entry_feat, indices, total),
                    entry_queue=_select(batch.entry_queue, indices, total))
    if variants is None:
        return small, None, indices, indices
    rows = torch.nonzero((variants.row_start >= start) & (variants.row_start < stop)).flatten()
    remap = {int(old): new for new, old in enumerate(rows)}
    sub = Variants(variants.row_start[rows] - start, variants.act_mask[rows],
                   variants.row_kind[rows], stop - start,
                   {s - start: remap[r] for s, r in variants.base_row.items() if start <= s < stop},
                   {(s - start, k): remap[r] for (s, k), r in variants.individual_rows.items()
                    if start <= s < stop},
                   [{**g, "start": g["start"] - start, "row": remap[g["row"]],
                     "flips": {k: remap[r] for k, r in g["flips"].items()}}
                    for g in variants.groups if start <= g["start"] < stop])
    return small, sub, indices, rows


def _slice_shape(value, shape, starts, rows, n_starts, n_rows):
    """Slice the sole batch axis; reject unrecognised stochastic/output geometry."""
    if tuple(value.shape) == tuple(shape):
        return value
    axes = [i for i, (a, b) in enumerate(zip(value.shape, shape)) if a != b]
    if len(value.shape) != len(shape) or len(axes) != 1:
        raise RuntimeError(f"LONGMEM cannot slice shape {tuple(value.shape)} to {tuple(shape)}")
    axis = axes[0]
    size = value.shape[axis]
    index = starts if size == n_starts else rows if size == n_rows else None
    if index is None or len(index) != shape[axis]:
        raise RuntimeError(f"LONGMEM unrecognised batch axis: {tuple(value.shape)} -> {tuple(shape)}")
    return value.index_select(axis, index.to(value.device))


class _RandomTape(TorchDispatchMode):
    """Record exact full-batch torch draws, then replay their start/variant slices.

    Includes private future generators and in-place Distribution.normal_ draws.
    Replay preserves common random numbers independently of microbatch shape.
    """
    names = {"rand", "randn", "randint", "rand_like", "randn_like", "randint_like",
             "normal", "normal_", "uniform_", "random_", "bernoulli", "bernoulli_"}

    def __init__(self, tape, selection=None, offload=False):
        super().__init__()
        self.tape, self.selection, self.offload, self.position = tape, selection, offload, 0
        self.controller_cells = []

    def release_controllers(self):
        # Checkpoint closures -> controller records -> autograd checkpoint graph
        # otherwise form a cycle retaining GPU tensors until Python's next GC.
        # These callbacks belong exclusively to the completed replay.
        for cell in self.controller_cells:
            cell.cell_contents = None
        self.controller_cells.clear()

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.release_controllers()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name.split("::")[-1]
        output = func(*args, **kwargs)
        if name not in self.names:
            return output
        if self.selection is None:
            saved = output.detach().clone()
            self.tape.append((str(func), saved.cpu() if self.offload else saved))
        else:
            if self.position >= len(self.tape) or self.tape[self.position][0] != str(func):
                raise RuntimeError("LONGMEM random call trace changed during microbatch replay")
            saved = self.tape[self.position][1]
            value = _slice_shape(saved, output.shape, *self.selection).to(output)
            if func._schema.is_mutable:
                output.copy_(value)
            else:
                output = value
        self.position += 1
        return output


def memory_imagine(trainer, batch, cfg, **kwargs):
    """Expose one full-batch forward and accumulate exact microbatch parameter VJPs."""
    from .dreamer import _restore_rng, _rng_state

    parameters = tuple(p for p in trainer.actor_parameters if p.requires_grad)
    structure = {}
    training = trainer._training_imagination

    class Replay(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            _trace_memory(trainer, 'full_forward_start')
            ctx.set_materialize_grads(False)
            ctx.params = params
            ctx.tape = []
            ctx.before = _rng_state(trainer.device)
            ctx.futures = {'items': []} if cfg.imagination_offload_futures else None
            from .parallel_futures import worker_count, parallel_forward
            if worker_count() > 1 and parallel_forward():
                if ctx.futures is None:
                    raise ValueError('parallel futures requires imagination_offload_futures')
                from .parallel_futures import forward_futures
                result, ctx.tape, ctx.futures = forward_futures(
                    trainer, batch, cfg, kwargs, ctx.before)
            else:
                with _RandomTape(ctx.tape, offload=cfg.imagination_offload_futures):
                    result = trainer.imagine(batch, **kwargs, _memory_replay=True, _memory_config=cfg,
                                             _memory_full_batch=batch, _memory_future_cache=ctx.futures)
            _trace_memory(trainer, 'full_forward_done', tape_draws=len(ctx.tape))
            leaves, spec = tree_flatten(result)
            ctx.spec = spec
            slots = [i for i, v in enumerate(leaves) if torch.is_tensor(v)]
            ctx.slots = slots
            structure.update(leaves=leaves, spec=spec, slots=slots)
            outputs = tuple(leaves[i] for i in slots)
            ctx.mark_non_differentiable(*(v for v in outputs if not v.is_floating_point()))
            return outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            after = _rng_state(trainer.device)
            old_training = trainer._training_imagination
            total = batch.state["s_m"].shape[0]
            n_rows = kwargs["variants"].n_rows if kwargs["variants"] is not None else total
            accumulated = [None] * len(ctx.params)
            try:
                trainer._training_imagination = training
                from .parallel_futures import worker_count, backward_futures
                if worker_count() > 1:
                    if ctx.futures is None:
                        raise ValueError('parallel futures requires imagination_offload_futures')
                    return backward_futures(trainer, batch, cfg, kwargs, ctx.tape,
                                            ctx.before, ctx.futures, ctx.slots, grad_outputs, ctx.spec)
                micro = cfg.imagination_microbatch or total
                for start in range(0, total, micro):
                    small, variants, starts, rows = _subset(
                        batch, kwargs["variants"], start, min(total, start + micro))
                    futures = range(len(ctx.futures['items'])) if ctx.futures is not None else (None,)
                    for future in futures:
                        _trace_memory(trainer, 'future_start', start=start, future=future)
                        replay = _RandomTape(ctx.tape, (starts, rows, total, n_rows))
                        _restore_rng(ctx.before, trainer.device)
                        spill = (_saved_tensor_spill(trainer)
                                 if cfg.imagination_offload_saved_tensors else nullcontext())
                        with torch.enable_grad(), replay, spill:
                            result = trainer.imagine(small, **{**kwargs, "variants": variants},
                                _memory_replay=True, _memory_config=cfg, _memory_full_batch=batch,
                                _memory_start_index=starts, _memory_row_index=rows,
                                _memory_future_cache=ctx.futures, _memory_active_future=future)
                            _trace_memory(trainer, 'future_forward_done', start=start, future=future)
                            leaves, _ = tree_flatten(result)
                            # Full-K topology/reductions are retained, with one future graph live.
                            outputs, vectors = [], []
                            for slot, vector in zip(ctx.slots, grad_outputs):
                                value = leaves[slot]
                                if vector is not None and value.requires_grad:
                                    outputs.append(value)
                                    vectors.append(_slice_shape(vector, value.shape, starts, rows,
                                                                total, n_rows))
                            if replay.position != len(ctx.tape):
                                raise RuntimeError("LONGMEM random call trace shortened during replay")
                            grads = torch.autograd.grad(outputs, ctx.params, vectors, allow_unused=True)
                            _trace_memory(trainer, 'future_vjp_done', start=start, future=future)
                        for i, grad in enumerate(grads):
                            if grad is not None:
                                accumulated[i] = grad if accumulated[i] is None else accumulated[i] + grad
                        del result, leaves, outputs, vectors, grads
            finally:
                trainer._training_imagination = old_training
                _restore_rng(after, trainer.device)
            return tuple(accumulated)

    outputs = Replay.apply(*parameters)
    leaves = structure.pop("leaves")
    for slot, value in zip(structure.pop("slots"), outputs):
        leaves[slot] = value
    result = tree_unflatten(leaves, structure.pop("spec"))
    if "action_audit" in result:
        # These are deliberately detached diagnostics on the original path.
        result["action_audit"] = {k: v.detach() for k, v in result["action_audit"].items()}
    return result


def _copy_runtime(value):
    # Preserve graph-bearing tensor identities; deepcopy cannot copy non-leaf tensors.
    if torch.is_tensor(value) or inspect.isfunction(value) or isinstance(value, torch.nn.Module):
        return value
    if isinstance(value, dict):
        return {k: _copy_runtime(v) for k, v in value.items()}
    if isinstance(value, list):
        # Controller records and entry logs are append-only. Copy the boundary
        # list, NOT every historical record at every segment (quadratic space).
        return list(value)
    if isinstance(value, tuple):
        return type(value)(*map(_copy_runtime, value)) if hasattr(value, "_fields") else tuple(
            _copy_runtime(v) for v in value)
    if type(value).__name__ == "LaneDwell":
        result = copy.copy(value)
        result.__dict__ = _copy_runtime(value.__dict__)
        return result
    # Trainer/config/callable objects are immutable for the duration of imagination.
    return value


def _controller_cells(controller):
    cells, seen = [], set()

    def visit(fn):
        if not inspect.isfunction(fn) or id(fn) in seen:
            return
        seen.add(id(fn))
        for cell in fn.__closure__ or ():
            if id(cell) not in seen:
                seen.add(id(cell))
                cells.append(cell)
                visit(cell.cell_contents)
    visit(controller)
    return cells


def segmented_rollout(model, initial, n_steps, segment, controller, kwargs):
    """Checkpoint continuation chunks, preserving WM history and controller/entry state.

    No state/history is detached. Recomputation rolls back callback and entry side
    effects, and reinstates runtime WM switches even after imagine's finally block.
    """
    from wms.model.driver_encoder import PARAMETER_NAMES

    signature = inspect.signature(model.rollout)
    if "k0" not in signature.parameters or "history" not in signature.parameters:
        raise ValueError("segmented imagination requires WM rollout continuation (k0/history)")
    cells = _controller_cells(controller)
    entries = kwargs.get("entries")
    carry = dict(initial)
    continuation = dict(kwargs)
    chunks = []
    tape = next((m for m in torch.utils._python_dispatch._get_current_dispatch_mode_stack()
                 if isinstance(m, _RandomTape)), None)
    if tape is not None:
        tape.controller_cells.extend(cells)

    def snapshot():
        values = [_copy_runtime(c.cell_contents) for c in cells]
        entry = None
        if entries is not None:
            entry = {k: (copy.deepcopy(v) if k == "process" else _copy_runtime(v))
                     for k, v in entries.__dict__.items()}
        return values, entry

    def restore(state):
        values, entry = state
        for cell, value in zip(cells, values):
            cell.cell_contents = _copy_runtime(value)
        if entry is not None:
            entries.__dict__.update({k: (copy.deepcopy(v) if k == "process" else _copy_runtime(v))
                                     for k, v in entry.items()})

    for offset in range(0, n_steps, segment):
        count = min(segment, n_steps - offset)
        # Each closure binds its OWN boundary, not the loop's eventual final values.
        def make_call(offset, count, continuation, boundary, position):
            switches = {k: getattr(model, k) for k in (
                "checkpoint_forward", "cav_exec_cap", "grad_decay_per_step") if hasattr(model, k)}

            @contextmanager
            def recompute_context():
                current = snapshot()
                old_switches = {k: getattr(model, k) for k in switches}
                old_position = tape.position if tape is not None else None
                restore(boundary)
                for k, v in switches.items():
                    setattr(model, k, v)
                if tape is not None:
                    tape.position = position
                try:
                    yield
                finally:
                    restore(current)
                    for k, v in old_switches.items():
                        setattr(model, k, v)
                    if tape is not None:
                        tape.position = old_position

            def call(state):
                return model.rollout(state, n_steps=count, k0=offset, cav_controller=controller,
                                     cav_teacher=None, truncation_steps=None, **continuation)
            return call, lambda: (nullcontext(), recompute_context())

        local = dict(continuation)
        for key in ("accel_noise", "lane_uniform", "state_noise"):
            if key in kwargs:
                local[key] = kwargs[key][:, offset:offset + count]
        call, contexts = make_call(offset, count, local, snapshot(),
                                   tape.position if tape is not None else 0)
        chunk = checkpoint(call, carry, use_reentrant=False, context_fn=contexts)
        chunks.append(chunk)
        carry = chunk["state"]
        continuation.update(history=chunk["history"], history_mask=chunk["history_mask"])
        for key in ("driver_history", "driver_history_mask", "driver_defaults"):
            continuation.pop(key, None)
        if "driver_parameters" in chunk:
            continuation["driver_params"] = {name: chunk["driver_parameters"][..., i]
                                              for i, name in enumerate(PARAMETER_NAMES)}
        if "driver_embedding" in chunk:
            continuation["driver_embedding"] = chunk["driver_embedding"]
    result = dict(chunks[-1])
    for key in ("positions", "speeds", "lanes", "masks", "accel", "entrant_seq"):
        if key in result:
            result[key] = torch.cat([c[key] for c in chunks], 1)
    return result


def memory_critic_loss(trainer, imagined, returns, scale):
    """Stream the independently mean-reduced imagined critic MSE and its gradient."""
    total = imagined["n_base"]
    micro = trainer.config.imagination_microbatch or total
    n_rows = imagined["returns"].shape[-1]
    from .dreamer import _detach, _rows
    states = [_detach(_rows(s, total, n_rows)) for member in imagined["states"] for s in member[:-1]]
    targets = returns.detach().reshape(-1, total)
    params = tuple(p for p in trainer.critic.parameters() if p.requires_grad)

    def pieces():
        for state, target in zip(states, targets):
            for start in range(0, total, micro):
                index = torch.arange(start, min(total, start + micro), device=target.device)
                yield _select(state, index, total), target[index]

    class CriticReplay(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *parameters):
            loss = returns.new_zeros(())
            for state, target in pieces():
                loss += ((trainer.critic(state) - target) / scale).square().sum() / returns.numel()
            return loss

        @staticmethod
        def backward(ctx, vector):
            accumulated = [None] * len(params)
            with torch.enable_grad():
                for state, target in pieces():
                    loss = ((trainer.critic(state) - target) / scale).square().sum() / returns.numel()
                    grads = torch.autograd.grad(loss, params, grad_outputs=vector, allow_unused=True)
                    for i, grad in enumerate(grads):
                        if grad is not None:
                            accumulated[i] = grad if accumulated[i] is None else accumulated[i] + grad
            return tuple(accumulated)
    return CriticReplay.apply(*params)
