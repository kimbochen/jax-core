from functools import lru_cache
from typing import *

import numpy as np

from autodidax.core import bind, full_raise, new_main_trace, get_aval
from autodidax.core import POP, PRIM_TOK
from autodidax.core import Trace, Tracer
from autodidax.jaxpr import jaxpr_as_fn, make_jaxpr, Jaxpr


# --- JVP Interpreter Classes ---

class JVPTracer(Tracer):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return get_aval(self.primal)

    def __repr__(self):
        return f'JVPTracer(primal={self.primal}, tangent={self.tangent})'


class JVPTrace(Trace):
    def to_tracer(self, val):
        aval = get_aval(val)
        tracer = JVPTracer(self, val, np.zeros_like(aval, aval.dtype))
        return tracer

    def pure(self, val):
        return self.to_tracer(val)

    def lift(self, val):
        return self.to_tracer(val)

    def apply_prim(self, prim, tracers, params):
        primals_in = [t.primal for t in tracers]
        tangents_in = [t.tangent for t in tracers]
        jvp_rule = JVP_RULES[prim]
        primal_out, tangent_out = jvp_rule(primals_in, tangents_in, **params)
        tracer_out = JVPTracer(self, primal_out, tangent_out)
        return tracer_out


# --- JVP Implementation Rules ---

def jvp_add(primals, tangents):
    ''' f = x + y -> f' = x' + y' '''
    (x, y), (x_dot, y_dot) = primals, tangents
    return x + y, x_dot + y_dot


def jvp_mul(primals, tangents):
    ''' f = x * y -> f' = x'y + xy' '''
    (x, y), (x_dot, y_dot) = primals, tangents
    return x * y, x_dot * y + x * y_dot


def jvp_sin(primals, tangents):
    ''' f = sin(x) -> f' = cos(x) * x' '''
    (x,), (x_dot,) = primals, tangents
    return POP.sin(x), POP.cos(x) * x_dot


def jvp_cos(primals, tangents):
    ''' f = cos(x) -> f' = -sin(x) * x' '''
    (x,), (x_dot,) = primals, tangents
    return POP.cos(x), -POP.sin(x) * x_dot


def jvp_neg(primals, tangents):
    ''' f = -x -> f' = -x' '''
    (x,), (x_dot,) = primals, tangents
    return POP.neg(x), POP.neg(x_dot)


def jvp_greater(primals, tangents):
    ''' Operator not differentiable '''
    x, y = primals
    primal_out = POP.greater(x, y)
    aval = get_aval(primal_out)
    tangent_out = np.zeros_like(aval, aval.dtype)
    return primal_out, tangent_out


def jvp_less(primals, tangents):
    ''' Operator not differentiable '''
    x, y = primals
    primal_out = POP.less(x, y)
    aval = get_aval(primal_out)
    tangent_out = np.zeros_like(aval, aval.dtype)
    return primal_out, tangent_out


def jvp_transpose(primals, tangents, *, perm):
    (primal,), (tangent,) = primals, tangents
    primal_out = POP.transpose(primal, perm)
    tangent_out = POP.transpose(tangent, perm)
    return primal_out, tangent_out


def jvp_broadcast(primals, tangents, *, shape, axes):
    (primal,), (tangent,) = primals, tangents
    primal_out = POP.broadcast(primal, shape, axes)
    tangent_out = POP.broadcast(tangent, shape, axes)
    return primal_out, tangent_out


def jvp_reduce_sum(primals, tangents, *, axis):
    (x,), (x_dot,) = primals, tangents
    primal_out = POP.reduce_sum(x, axis)
    tangent_out = POP.reduce_sum(x_dot, axis)
    return primal_out, tangent_out


def jvp_xla_call(primals, tangents, *, jaxpr, num_consts):
    del num_consts
    new_jaxpr, new_consts = jvp_jaxpr(jaxpr)
    primals_out, tangents_out = bind(
        PRIM_TOK.xla_call, *new_consts, *primals, *tangents,
        jaxpr=new_jaxpr, num_consts=len(new_consts)
    )
    return primals_out, tangents_out


@lru_cache
def jvp_jaxpr(jaxpr: Jaxpr) -> Tuple[Jaxpr, List[Any]]:
    def traceable_jvp(*inputs):
        n = len(inputs) // 2
        primals, tangents = inputs[:n], inputs[n:]
        traced_jvp = jvp(jaxpr_as_fn(jaxpr), primals, tangents)
        return traced_jvp

    avals_in = [v.aval for v in jaxpr.in_binders]  # Primals and tangents are both this
    new_jaxpr, new_consts = make_jaxpr(traceable_jvp, *avals_in, *avals_in)

    return new_jaxpr, new_consts


JVP_RULES = {
    PRIM_TOK.add: jvp_add,
    PRIM_TOK.mul: jvp_mul,
    PRIM_TOK.neg: jvp_neg,
    PRIM_TOK.sin: jvp_sin,
    PRIM_TOK.cos: jvp_cos,
    PRIM_TOK.greater: jvp_greater,
    PRIM_TOK.less: jvp_less,
    PRIM_TOK.transpose: jvp_transpose,
    PRIM_TOK.reduce_sum: jvp_reduce_sum,
    PRIM_TOK.xla_call: jvp_xla_call
}


# --- JVP API ---

def jvp(fn, primals, tangents):
    to_tuple = lambda x: x if isinstance(x, tuple) else (x,)
    primals, tangents = map(to_tuple, (primals, tangents))

    with new_main_trace(JVPTrace) as main_trace:
        trace = JVPTrace(main_trace)
        in_tracers = [JVPTracer(trace, p, t) for p, t in zip(primals, tangents)]
        fn_outs = fn(*in_tracers)
        if isinstance(fn_outs, (tuple, list)):
            out_primals_tangents = [lower_fn_output(trace, fout) for fout in fn_outs]
            out_primals, out_tangents = zip(*out_primals_tangents)
        else:
            out_primals, out_tangents = lower_fn_output(trace, fn_outs)

    return out_primals, out_tangents


def lower_fn_output(trace, fn_out):
    out_tracer = full_raise(trace, fn_out)
    out_primal, out_tangent = out_tracer.primal, out_tracer.tangent
    return out_primal, out_tangent
