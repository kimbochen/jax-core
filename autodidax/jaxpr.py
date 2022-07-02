import string
import itertools as it
from collections import defaultdict
from functools import lru_cache
from operator import is_
from typing import *

from autodidax.core import bind, full_raise, new_dynamic_trace, new_main_trace
from autodidax.core import get_aval, ShapedArray
from autodidax.core import POP, PRIM_TOK, JAX_TYPES
from autodidax.core import Primitive, Trace, Tracer


# --- Data Structures ---

class Var:
    '''
    Jaxpr Variable.
    var ::= a | b | c | ...
    '''
    def __init__(self, aval: ShapedArray):
        self.aval = aval

    def __repr__(self):
        return f'Var(aval={self.aval})'


class Literal:
    '''
    Jaxpr Literal.
    literal ::= <int32> | <int64> | <float32> | <float64>
    '''
    def __init__(self, val):
        aval = get_aval(val)
        self.val: Any = aval.val
        self.aval: ShapedArray = ShapedArray(aval.shape, aval.dtype)

    def __repr__(self):
        return f'Literal(val={self.val}, aval={self.aval})'


# Jaxpr Atom.
# atom ::= <var> | <literal>
Atom = Union[Var, Literal]


class JaxprEqn(NamedTuple):
    '''
    Jaxpr Equation.
    eqn ::= <binder> , ... = <primitive> [ <params> ] <atom> , ...
    binder ::= <var>:<array_type>
    array_type ::= <dtype>[<shape>]
    '''
    prim: Primitive
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binder: Var

    def __repr__(self):
        inputs_str = ', '.join(f'{a}' for a in self.inputs)
        params_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'JaxprEqn(prim={self.prim.name}[{params_str}], inputs=[{inputs_str}], out_binder={self.out_binder})'


class Jaxpr(NamedTuple):
    '''
    Jaxpr.
    jaxpr ::= {
        lambda <binder> , ... .
        let <eqn>
            ...
        in ( <atom> , ... ) 
    }
    '''
    in_binders: List[Var]
    eqns: List[JaxprEqn]
    outputs: List[Atom]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return is_(self, other)

    def __str__(self):
        name_gen = (
            ''.join(s)
            for r in it.count(1)
            for s in it.permutations(string.ascii_lowercase, r)
        )
        var_name = defaultdict(lambda: next(name_gen))  # Generates a new name if key not found

        in_binders_str = ', '.join(f'{var_name[v]}:{v.aval}' for v in self.in_binders)
        eqns_str = '\n    '.join(
            f'{var_name[eqn.out_binder]}:{eqn.out_binder.aval} = '\
            f'{eqn.prim.name}[{" ".join(f"{k}={v}" for k, v in sorted(eqn.params.items()))}] '\
            f'{" ".join(var_name[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs)}'
            for eqn in self.eqns
        )
        outputs_str = ', '.join(f'{var_name[v]}' for v in self.outputs)

        return f'{{\nlambda {in_binders_str} .\nlet {eqns_str}\nin ( {outputs_str} )\n}}'


# --- Helper Functions ---

raise_to_shaped = lambda aval: ShapedArray(aval.shape, aval.dtype)
raise_val_to_shaped = lambda val: raise_to_shaped(get_aval(val))


# --- Abstract Evaluation Rules ---

uniop_abstract_eval = lambda x: x


def binop_abstract_eval(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
    assert isinstance(x, ShapedArray) and isinstance(y, ShapedArray)
    if x != y:
        raise TypeError(f'Incompatible ShapedArray: {x}, {y}')
    else:
        return x


def compare_abstract_eval(x: ShapedArray, y: ShapedArray) -> ShapedArray:
    assert isinstance(x, ShapedArray) and isinstance(y, ShapedArray)
    if x.shape != y.shape:
        raise TypeError(f'Incompatible ShapedArray: {x}, {y}')
    else:
        return ShapedArray(x.shape, np.bool_)


def transpose_abstract_eval(x: ShapedArray, *, perm: Sequence[int]) -> ShapedArray:
    transposed_shape = [x.shape[idx] for idx in perm]
    return ShapedArray(transposed_shape, x.dtype)


def broadcast_abstract_eval(
    x: ShapedArray, *, shape: Sequence[int], axes: Sequence[int]
) -> ShapedArray:
    return ShapedArray(tuple(shape), x.dtype)


def reduce_sum_abstract_eval(x: ShapedArray, *, axis: Tuple[int, ...]) -> ShapedArray:
    reduced_axes = set(axis)
    reduced_shape = [d for i, d in enumerate(x.shape) if i not in reduced_axes]
    return ShapedArray(tuple(reduced_shape), x.dtype)


ABSTRACT_EVAL_RULES = {
    PRIM_TOK.add: binop_abstract_eval,
    PRIM_TOK.mul: binop_abstract_eval,
    PRIM_TOK.neg: uniop_abstract_eval,
    PRIM_TOK.sin: uniop_abstract_eval,
    PRIM_TOK.cos: uniop_abstract_eval,
    PRIM_TOK.greater: compare_abstract_eval,
    PRIM_TOK.less: compare_abstract_eval,
    PRIM_TOK.transpose: transpose_abstract_eval,
    PRIM_TOK.broadcast: broadcast_abstract_eval,
    PRIM_TOK.reduce_sum: reduce_sum_abstract_eval
}


# --- Jaxpr Interpreter Classes ---

class JaxprTracer(Tracer):
    __slots__ = ['aval']

    def __init__(self, trace, aval: ShapedArray):
        self._trace = trace
        self.aval = aval

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'JaxprTracer(aval={self.aval})'


class JaxprTrace(Trace):
    def new_tracer(self, aval):
        tracer = JaxprTracer(self, aval)
        self.builder.tracers.append(tracer)
        return tracer

    def make_const_tracer(self, val: Any) -> JaxprTracer:
        tracer = self.builder.get_const(val)
        if tracer is None:
            aval = raise_val_to_shaped(val)
            tracer = self.new_tracer(aval)
            self.builder.add_const(tracer, val)
        return tracer

    def pure(self, val):
        return self.make_const_tracer(val)

    def lift(self, val):
        return self.make_const_tracer(val)

    def new_arg(self, val: Any) -> JaxprTracer:
        aval = raise_to_shaped(val)
        tracer = self.new_tracer(aval)
        self.builder.add_var(tracer)
        return tracer

    def apply_prim(self, prim, tracers, params):
        avals_in = [t.aval for t in tracers]
        abstract_eval_rule = ABSTRACT_EVAL_RULES[prim]
        aval_out = abstract_eval_rule(*avals_in, **params)
        tracer_out = self.new_tracer(aval_out)

        inputs = [self.builder.tracer_var[t] for t in tracers]
        var_out = self.builder.add_var(tracer_out)
        eqn = JaxprEqn(prim, inputs, params, var_out)
        self.builder.eqns.append(eqn)

        return tracer_out

    @property
    def builder(self):
        return self.main.global_data


# ---------- Jaxpr Builder ----------

class JaxprBuilder:
    def __init__(self):
        self.tracer_var: Dict[JaxprTracer, Var] = dict()
        self.const_tracer: Dict[int, JaxprTracer] = dict()
        self.const_val: Dict[Var, Any] = dict()
        self.tracers: List[JaxprTracer] = []
        self.eqns: List[JaxprEqn] = []

    def add_var(self, tracer: JaxprTracer) -> Var:
        assert tracer not in self.tracer_var
        var = Var(tracer.aval)
        self.tracer_var[tracer] = var
        return var

    def get_const(self, val: Any) -> JaxprTracer:
        return self.const_tracer.get(id(val))

    def add_const(self, tracer: JaxprTracer, val: ShapedArray) -> None:
        var = self.add_var(tracer)
        self.const_tracer[id(val)] = tracer
        self.const_val[var] = val

    def build(
        self, in_tracers: List[JaxprTracer], out_tracers: List[JaxprTracer]
    ) -> Tuple[Jaxpr, List[Any]]:
        in_tracer_vars = [self.tracer_var[t] for t in in_tracers]
        const_vars = list(self.const_val.keys())
        in_binders = const_vars + in_tracer_vars
        outputs = [self.tracer_var[t] for t in out_tracers]

        jaxpr = Jaxpr(in_binders, self.eqns, outputs)
        typecheck_jaxpr(jaxpr)
        jaxpr, const_vals = _inline_literals(jaxpr, self.const_val.values())

        return jaxpr, const_vals


def partition_list(pos_filter, full_list):
    pos_list = filter(pos_filter, full_list)
    neg_list = filter(lambda e: not pos_filter(e), full_list)
    return list(pos_list), list(neg_list)


def _inline_literals(jaxpr: Jaxpr, consts: List[Any]) -> Tuple[Jaxpr, List[Any]]:
    const_binders, other_binders = jaxpr.in_binders[:len(consts)], jaxpr.in_binders[len(consts):]

    is_scalar = lambda x: (type(x) in JAX_TYPES) and (get_aval(x).shape is None)
    new_const_binders, lit_binders = partition_list(is_scalar, const_binders)
    new_consts, lit_vals = partition_list(is_scalar, consts)

    literals = dict(zip(lit_binders, map(Literal, lit_vals)))
    eqn_inputs = lambda eqn: [literals.get(x, x) for x in eqn.inputs]

    new_eqns = [
        JaxprEqn(eqn.prim, eqn_inputs(eqn), eqn.params, eqn.out_binder)
        for eqn in jaxpr.eqns
    ]
    new_outs = [literals.get(x, x) for x in jaxpr.outputs]
    new_jaxpr = Jaxpr(new_const_binders + other_binders, new_eqns, new_outs)
    typecheck_jaxpr(new_jaxpr)

    return new_jaxpr, new_consts


# ---------- Typecheck ----------

class JaxprType(NamedTuple):
    in_types: List[ShapedArray]
    out_types: List[ShapedArray]

    def __repr__(self):
        in_types = ', '.join(map(str, self.in_types))
        out_types = ', '.join(map(str, self.out_types))
        return f'({in_types}) -> ({out_types})'


def typecheck_jaxpr(jaxpr: Jaxpr) -> JaxprType:
    env: Set[Var] = set(jaxpr.in_binders)
    assert len(env) == len(jaxpr.in_binders), 'Duplicated variables.'

    for eqn in jaxpr.eqns:
        in_types = [typecheck_atom(env, x) for x in eqn.inputs]
        abstract_eval_rule = ABSTRACT_EVAL_RULES[eqn.prim]
        out_type = abstract_eval_rule(*in_types, **eqn.params)
        if out_type != eqn.out_binder.aval:
            raise TypeError(
                f'Evaluation has {out_type}, but {out_binder.aval} is traced.'
            )
        if eqn.out_binder in env:
            raise TypeError(f'Duplicated binder {out_binder}')
        else:
            env.add(eqn.out_binder)

    in_types = [v.aval for v in jaxpr.in_binders]
    out_types = [typecheck_atom(env, x) for x in jaxpr.outputs]

    return JaxprType(in_types, out_types)


def typecheck_atom(env: Set[Var], x: Atom) -> ShapedArray:
    if isinstance(x, Var):
        if x not in env:
            raise TypeError(f'Unbound variable of type {type(x)}')
        return x.aval
    elif isinstance(x, Literal):
        return raise_val_to_shaped(x.val)
    else:
        raise TypeError(f'Invalid type {type(x)}')


# --- Jaxpr API ---

@lru_cache()
def make_jaxpr(fn, *avals_in):
    to_iterable = lambda x: x if isinstance(x, (tuple, list)) else (x,)
    builder = JaxprBuilder()

    with new_main_trace(JaxprTrace, builder) as main_trace:
        with new_dynamic_trace(main_trace):  # omnistaging
            trace = JaxprTrace(main_trace)
            in_tracers = list(map(trace.new_arg, avals_in))
            outs = fn(*in_tracers)
            out_tracers = [full_raise(trace, out) for out in to_iterable(outs)]
            jaxpr, consts = builder.build(in_tracers, out_tracers)

    return jaxpr, consts


def eval_jaxpr(jaxpr: Jaxpr, args: List[Any]) -> List[Any]:
    env: Dict[Var, Any] = dict()

    def read(x: Atom):
        return env[x] if isinstance(x, Var) else x.val

    def write(binder: Var, val: Any):
        assert binder not in env  # Single-assignment of variable to value
        env[binder] = val

    list(map(write, jaxpr.in_binders, args))
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.inputs)
        out = bind(eqn.prim, *in_vals, **eqn.params)
        write(eqn.out_binder, out)
    out_vals = list(map(read, jaxpr.outputs))

    return out_vals if len(out_vals) > 1 else out_vals[0]


def jaxpr_as_fn(jaxpr: Jaxpr):
    fn = lambda *args: eval_jaxpr(jaxpr, args)
    return fn
