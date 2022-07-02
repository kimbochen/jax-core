from functools import lru_cache, partial
from typing import *

import numpy as np
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc


from autodidax.core import bind, EVAL_RULES
from autodidax.core import JAX_TYPES, POP, PRIM_TOK
from autodidax.core import ShapedArray, Trace, Tracer
from autodidax.jaxpr import make_jaxpr, raise_val_to_shaped, typecheck_jaxpr
from autodidax.jaxpr import Atom, Var, Jaxpr


xe = xc._xla
xops = xc._xla.ops


# --- XLA Call Evaluation Rule ---

class IDHashable:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return id(self.val)

    def __eq__(self, other):
        return isinstance(other, IDHashable) and id(self.val) == id(other.val)


def xla_call_eval(*args, jaxpr: Jaxpr, num_consts: int):
    consts, args = args[:num_consts], args[num_consts:]
    hashable_consts = tuple(map(IDHashable, consts))
    execute = make_xla_callable(IDHashable(jaxpr), hashable_consts)
    return execute(*args)

EVAL_RULES[PRIM_TOK.xla_call] = xla_call_eval


@lru_cache()
def make_xla_callable(hashable_jaxpr: IDHashable, hashable_consts: Tuple[IDHashable]):
    jaxpr: Jaxpr = hashable_jaxpr.val
    typecheck_jaxpr(jaxpr)

    consts = [x.val for x in hashable_consts]
    avals_in = [v.aval for v in jaxpr.in_binders[len(consts):]]
    builder = xc.XlaBuilder('xla_call')
    xla_consts = get_xla_consts(builder, consts)
    xla_params = get_xla_params(builder, avals_in)

    outs = jaxpr_subcomp(builder, jaxpr, xla_consts + xla_params)
    xla_outs = xops.Tuple(builder, outs)
    compiled_fn = xb.get_backend(None).compile(builder.build(xla_outs))

    return partial(execute_compiled_fn, compiled_fn, [v.aval for v in jaxpr.outputs])


def jaxpr_subcomp(builder: xe.XlaBuilder, jaxpr: Jaxpr, args: List[xe.XlaOp]) -> xe.XlaOp:
    env: Dict[Var, xe.XlaOp] = dict()

    def read(x: Atom) -> xe.XlaOp:
        return env[x] if isinstance(x, Var) else xops.Constant(builder, np.asarray(x.val))

    env.update(dict(zip(jaxpr.in_binders, args)))
    for eqn in jaxpr.eqns:
        avals_in = [x.aval for x in eqn.inputs]
        vals_in = list(map(read, eqn.inputs))
        translate = XLA_TRANSLATION_RULES[eqn.prim]
        val_out = translate(builder, avals_in, vals_in, **eqn.params)
        env[eqn.out_binder] = val_out
    outs = list(map(read, jaxpr.outputs))

    return outs


XLA_TYPES = (bool, int, float, np.ndarray, np.float64, np.float32)
INPUT_HANDLERS = dict.fromkeys(XLA_TYPES, xb.get_backend(None).buffer_from_pyval)


def execute_compiled_fn(compiled_fn, out_avals, *args):
    input_bufs = [INPUT_HANDLERS[type(x)](x) for x in args]
    output_bufs = compiled_fn.execute(input_bufs)
    results = [handle_result(aval, buf) for aval, buf in zip(out_avals, output_bufs)]
    return results if len(results) > 1 else results[0]


# --- Device-Memory-Persistent Arrays ---

class DeviceArray:
    def __init__(self, aval: ShapedArray, buf: Any):
        self.aval = aval
        self.buf = buf

    dtype = property(lambda self: self.aval.dtype)
    shape = property(lambda self: self.aval.shape)
    ndim  = property(lambda self: self.aval.ndim)

    def __array__(self):
        return self.buf.to_py()

    def __repr__(self):
        return repr(self.buf.to_py())

    def __str__(self):
        return str(self.buf.to_py())

    _add = staticmethod(POP.add)
    _radd = staticmethod(POP.add)
    _mul = staticmethod(POP.mul)
    _rmul = staticmethod(POP.mul)
    _neg = staticmethod(POP.neg)
    _gt = staticmethod(POP.greater)
    _lt = staticmethod(POP.less)

INPUT_HANDLERS[DeviceArray] = lambda x: x.buf
JAX_TYPES.add(DeviceArray)


def handle_result(aval: ShapedArray, buf):
    return DeviceArray(aval, buf)


# --- XLA Translation Rule ---

def direct_translation(xla_op, builder, avals_in, vals_in):
    del builder, avals_in
    return xla_op(*vals_in)


def broadcast_translation(builder, avals_in, vals_in, *, shape, axes):
    (x,) = vals_in
    out_dims = [ax for ax in range(len(shape)) if ax not in axes]
    return xops.BroadcastInDim(x, shape, out_dims)


def reduce_sum_translation(builder, avals_in, vals_in, *, axis):
    (aval_x,), (x,) = avals_in, vals_in
    zero = xops.ConstantLiteral(builder, np.array(0, aval_x.dtype))
    sub_builder = xc.XlaBuilder('add')
    shape = get_xla_shape(ShapedArray((), aval_x.dtype))
    xops.Add(xops.Parameter(sub_builder, 0, shape), xops.Parameter(sub_builder, 1, shape))
    return xops.Reduce(builder, [x], [zero], sub_builder.build(), axis)


def xla_call_translation(builder, avals_in, vals_in, *, jaxpr, num_consts):
    del num_consts
    sub_builder = xc.XlaBuilder('inner xla_call')
    xla_params = get_xla_params(sub_builder, avals_in)
    outputs = jaxpr_subcomp(sub_builder, jaxpr, xla_params)
    sub_builder = sub_builder.build(xops.Tuple(sub_builder, outputs))
    return destructure_tuple(builder, xops.Call(builder, sub_builder, vals_in))


XLA_TRANSLATION_RULES = {
    PRIM_TOK.add: partial(direct_translation, xops.Add),
    PRIM_TOK.mul: partial(direct_translation, xops.Mul),
    PRIM_TOK.neg: partial(direct_translation, xops.Neg),
    PRIM_TOK.sin: partial(direct_translation, xops.Sin),
    PRIM_TOK.cos: partial(direct_translation, xops.Cos),
    PRIM_TOK.greater: partial(direct_translation, xops.Gt),
    PRIM_TOK.less: partial(direct_translation, xops.Lt),
    PRIM_TOK.broadcast: broadcast_translation,
    PRIM_TOK.reduce_sum: reduce_sum_translation,
    PRIM_TOK.xla_call: xla_call_translation
}


# ---------- XLA Call Helper Functions ----------

def get_xla_consts(builder: xe.XlaBuilder, consts: List[Any]) -> List[xe.XlaOp]:
    unique_consts = set(consts)
    xla_consts = {const: xops.ConstantLiteral(builder, const) for const in unique_consts}
    return [xla_consts[const] for const in consts]


def get_xla_params(builder: xe.XlaBuilder, avals: List[ShapedArray]) -> List[xe.XlaOp]:
    xla_param = lambda idx, aval: xops.Parameter(builder, idx, get_xla_shape(aval))
    return [xla_param(idx, aval) for idx, aval in enumerate(avals)]


def get_xla_shape(aval: ShapedArray) -> xe.Shape:
    return xc.Shape.array_shape(xc.dtype_to_etype(aval.dtype), aval.shape)


def destructure_tuple(builder, tup):
    num_elem = len(builder.get_shape(tup).tuple_shapes())
    return [xops.GetTupleElement(tup, i) for i in range(num_elem)]


# ---------- JIT API ----------

def jit(fn):
    def jitted_fn(*args):
        avals_in = [raise_val_to_shaped(arg) for arg in args]
        jaxpr, consts = make_jaxpr(fn, *avals_in)
        outputs = bind(PRIM_TOK.xla_call, *consts, *args, jaxpr=jaxpr, num_consts=len(consts))
        return outputs
    return jitted_fn
