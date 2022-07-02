from functools import lru_cache, partial
from typing import *

import numpy as np

from autodidax.core import bind, full_raise, new_main_trace
from autodidax.core import get_aval, ShapedArray
from autodidax.core import POP, PRIM_TOK
from autodidax.core import Trace, Tracer
from autodidax.jaxpr import jaxpr_as_fn, make_jaxpr, Jaxpr


# --- Batching Interpreter Classes ---

class SentinelDim:
    def __repr__(self):
        return 'SentinelDim'

BatchDim = Union[SentinelDim, int]
UNMAPPED = SentinelDim()


class BatchTracer(Tracer):
    def __init__(self, trace, val, batch_dim: BatchDim):
        self._trace = trace
        self.val = val
        self.batch_dim = batch_dim

    @property
    def aval(self):
        aval = get_aval(self.val)
        if self.batch_dim is not UNMAPPED:
            shape = aval.shape
            unbatched_shape = shape[:self.batch_dim] + shape[self.batch_dim+1:]
            return ShapedArray(unbatched_shape, aval.dtype)
        else:
            return aval

    def full_lower(self):
        return full_lower(self.val) if self.batch_dim is UNMAPPED else self

    def __repr__(self):
        return f'BatchTracer(val={self.val}, batch_dim={self.batch_dim})'


class BatchTrace(Trace):
    def to_tracer(self, val):
        return BatchTracer(self, val, UNMAPPED)

    def pure(self, val):
        return self.to_tracer(val)

    def lift(self, val):
        return self.to_tracer(val)

    def apply_prim(self, prim, tracers, params):
        vals_in = [t.val for t in tracers]
        bdims_in = [t.batch_dim for t in tracers]
        vmap_rule = VMAP_RULES[prim]
        val_out, bdim_out = vmap_rule(self.axis_size, vals_in, bdims_in, **params)
        tracer_out = BatchTracer(self, val_out, bdim_out)
        return tracer_out

    @property
    def axis_size(self):
        return self.main.global_data

    def __repr__(self):
        return f'BatchTrace(axis_size={self.axis_size})'


# --- Batching Rules ---

def binop_batching_rule(prim_op, axis_size, vals_in, bdims_in):
    (x, y), (bdim_x, bdim_y) = vals_in, bdims_in
    if bdim_x != bdim_y:
        if bdim_x is UNMAPPED:
            x = add_batch_axis(x, bdim_y, axis_size)
            return prim_op(x, y), bdim_y
        elif bdim_y is UNMAPPED:
            y = add_batch_axis(y, bdim_x, axis_size)
            return prim_op(x, y), bdim_x
        else:
            x = align_batch_axis(x, bdim_x, bdim_y)
            return prim_op(x, y), bdim_y
    else:
        return prim_op(x, y), bdim_y


def uniop_batching_rule(prim_op, axis_size, vals_in, bdims_in):
    (x,), (bdim_x,) = vals_in, bdims_in
    return prim_op(x), bdim_x


def add_batch_axis(z, bdim, axis_size):
    tgt_shape = list(np.shape(z))
    tgt_shape.insert(bdim, axis_size)
    z = POP.broadcast(z, tgt_shape, [bdim])
    return z


def align_batch_axis(z, bdim_z, bdim):
    if bdim_z == bdim:
        return z
    perm = list(range(np.ndim(z)))
    perm[bdim_z], perm[bdim] = perm[bdim], perm[bdim_z]
    z = POP.transpose(z, perm)
    return z


def reduce_sum_batching_rule(axis_size, vals_in, bdims_in, *, axis):
    (x,), (bdim,) = vals_in, bdims_in
    batched_axis = (ax + 1 if ax >= bdim else ax for ax in axis)
    out_bdim = bdim - len([ax for ax in axis if ax < bdim])
    return POP.reduce_sum(x, batched_axis), out_bdim


def xla_call_batching_rule(axis_size, vals_in, bdims_in, *, jaxpr, num_consts):
    del num_consts
    (bdim_in,) = bdims_in
    new_jaxpr, new_consts = vmap_jaxpr(jaxpr, axis_size, bdim_in)
    output = bind(
        PRIM_TOK.xla_call, *new_consts, *vals_in,
        jaxpr=new_jaxpr, num_consts=len(new_consts)
    )
    return output, 0


@lru_cache
def vmap_jaxpr(
    jaxpr: Jaxpr, axis_size: int, bdim_in: Tuple[BatchDim, ...]
) -> Tuple[Jaxpr, List[Any]]:
    traceable_vmap = vmap(jaxpr_as_fn(jaxpr), bdim_in)
    avals_in = [unmapped_aval(axis_size, bdim_in, v.aval) for v in jaxpr.in_binders]
    new_jaxpr, new_consts = make_jaxpr(traceable_vmap, *avals_in)
    return new_jaxpr, new_consts


def unmapped_aval(axis_size: int, batch_dim: BatchDim, aval: ShapedArray) -> ShapedArray:
    if batch_dim is not UNMAPPED:
        shape = list(aval.shape)
        shape.insert(batch_dim, axis_size)
        return ShapedArray(tuple(shape), aval.dtype)
    else:
        return aval


VMAP_RULES = {
    PRIM_TOK.add: partial(binop_batching_rule, POP.add),
    PRIM_TOK.mul: partial(binop_batching_rule, POP.mul),
    PRIM_TOK.greater: partial(binop_batching_rule, POP.greater),
    PRIM_TOK.less: partial(binop_batching_rule, POP.less),
    PRIM_TOK.neg: partial(uniop_batching_rule, POP.neg),
    PRIM_TOK.sin: partial(uniop_batching_rule, POP.sin),
    PRIM_TOK.cos: partial(uniop_batching_rule, POP.cos),
    PRIM_TOK.reduce_sum: reduce_sum_batching_rule,
    PRIM_TOK.xla_call: xla_call_batching_rule
}


# --- vmap API ---

def vmap(fn, in_axis):
    def batched_fn(*args):
        axis_size = args[0].shape[in_axis]  # Only 1 common batch dimension is supported
        with new_main_trace(BatchTrace, axis_size) as main_trace:
            trace = BatchTrace(main_trace)
            in_tracers = [BatchTracer(trace, arg, in_axis) for arg in args]
            fn_outs = fn(*in_tracers)
            if isinstance(fn_outs, (tuple, list)):
                outputs = [
                    lower_fn_output(trace, axis_size, fn_out) for fn_out in fn_outs
                ]
            else:
                outputs = lower_fn_output(trace, axis_size, fn_outs)
        return outputs
    return batched_fn


def lower_fn_output(trace, axis_size, fn_out):
    out_tracer = full_raise(trace, fn_out)
    batch_dim = out_tracer.batch_dim
    if batch_dim is UNMAPPED:
        return add_batch_axis(out_tracer.val, 0, axis_size)
    else:
        return align_batch_axis(out_tracer.val, batch_dim, 0)
