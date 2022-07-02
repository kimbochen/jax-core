from typing import *
from weakref import ref, ReferenceType

from autodidax.auto_diff import jvp
from autodidax.core import bind, full_raise, full_lower, get_aval, new_main_trace
from autodidax.core import Primitive, ShapedArray, Trace, Tracer
from autodidax.jaxpr import raise_to_shaped, raise_val_to_shaped
from autodidax.jaxpr import eval_jaxpr, typecheck_jaxpr
from autodidax.jaxpr import ABSTRACT_EVAL_RULES, Jaxpr, JaxprEqn, Var


class PartialVal(NamedTuple):
    aval: ShapedArray
    const: Optional[Any] = None

    @classmethod
    def known(cls, val: Any):
        aval = get_aval(val)
        return cls(aval, val)

    @classmethod
    def unknown(cls, aval: ShapedArray):
        return cls(aval)

    is_known = property(lambda self: self.const is not None)
    is_unknown = property(lambda self: self.const is None)


class LambdaBindingRecipe(NamedTuple):
    pass


class ConstRecipe(NamedTuple):
    val: Any


class JaxprEqnRecipe(NamedTuple):
    prim: Primitive
    tracers_in: List['PartialEvalTracer']
    params: Dict[str, Any]
    aval_out: ShapedArray
    tracer_ref_out: 'ReferenceType[PartialEvalTracer]'


JaxprRecipe = Union[LambdaBindingRecipe, ConstRecipe, JaxprEqnRecipe]


class PartialEvalTracer(Tracer):
    def __init__(self, trace, pval: PartialVal, recipe: Optional[JaxprRecipe] = None):
        self._trace = trace
        self.pval = pval
        self.recipe = recipe

    @property
    def aval(self):
        return self.pval.aval

    def full_lower(self):
        return full_lower(self.pval.const) if self.pval.is_known else self

    def __hash__(self):
        return id(self)


class PartialEvalTrace(Trace):
    def new_known_pval(self, val: Any):
        return PartialEvalTracer(self, PartialVal.known(val))
    pure = lift = new_known_pval

    def new_arg(self, pval: PartialVal) -> PartialEvalTracer:
        return PartialEvalTracer(self, pval, LambdaBindingRecipe())

    def new_const(self, tracer: PartialEvalTracer) -> PartialEvalTracer:
        if tracer.pval.is_known:
            pval = PartialVal.unknown(raise_to_shaped(tracer.aval))
            const_recipe = ConstRecipe(tracer.pval.const)
            return PartialEvalTracer(self, pval, const_recipe)
        else:
            return tracer

    def apply_prim(self, prim, tracers, params):
        if all(t.pval.is_known for t in tracers):
            args = list(map(full_lower, tracers))
            return bind(prim, *args, **params)

        partial_eval_rule = PARTIAL_EVAL_RULES.get(prim)

        if partial_eval_rule is None:
            tracers_in = [self.new_const(t) for t in tracers]
            avals_in = [t.aval for t in tracers_in]

            abstract_eval_rule = ABSTRACT_EVAL_RULES[prim]
            aval_out = abstract_eval_rule(*avals_in, **params)

            tracer_out = PartialEvalTracer(self, PartialVal.unknown(aval_out))
            tracer_ref_out = ref(tracer_out)
            eqn_recipe = JaxprEqnRecipe(prim, tracers_in, params, aval_out, tracer_ref_out)
            tracer_out.recipe = eqn_recipe

            return tracer_out
        else:
            return partial_eval_rule(self, tracers, **params)


PARTIAL_EVAL_RULES = {}


def tracers_to_jaxpr(tracers_in: List[PartialEvalTracer], tracers_out: List[PartialEvalTracer]):
    tracer2var: Dict[PartialEvalTracer, Var] = {t: Var(raise_to_shaped(t)) for t in tracers_in}
    constvar2val: Dict[Var, Any] = dict()
    constid2var: Dict[int, Var] = dict()
    processed_eqns: Set[int] = set()
    eqns: List[JaxprEqn] = []

    for t in toposort(tracers_out, get_tracer_parents):
        if isinstance(t.recipe, LambdaBindingRecipe):
            assert t in tracer2var
        elif isinstance(t.recipe, ConstRecipe):
            val = t.recipe.val
            var = constid2var.get(id(val))
            if var is None:
                aval = raise_val_to_shaped(val)
                var = constid2var[id(val)] = Var(aval)
                constvar2val[var] = val
            tracer2var[t] = var
        elif isinstance(t.recipe, JaxprEqnRecipe):
            if id(t.recipe) not in processed_eqns:
                eqns.append(recipe_to_eqn(tracer2var, t.recipe))
                processed_eqns.add(id(t.recipe))
        else:
            raise TypeError(t.recipe)

    constvars, constvals = list(constvar2val.keys()), list(constvar2val.values())
    in_binders = constvars + [tracer2var[t] for t in tracers_in]
    out_vars = [tracer2var[t] for t in tracers_out]
    jaxpr = Jaxpr(in_binders, eqns, out_vars)
    typecheck_jaxpr(jaxpr)

    return jaxpr, constvals


def recipe_to_eqn(
    tracer2var: Dict[PartialEvalTracer, Var], recipe: JaxprEqnRecipe
) -> JaxprEqn:
    inputs = [tracer2var[t] for t in recipe.tracers_in]
    out_binder = Var(recipe.aval_out)
    t_ref = recipe.tracer_ref_out
    if t_ref() is not None:
        tracer2var[t_ref()] = out_binder
    return JaxprEqn(recipe.prim, inputs, recipe.params, out_binder)


def get_tracer_parents(t: PartialEvalTracer) -> List[PartialEvalTracer]:
    return t.recipe.tracers_in if isinstance(t.recipe, JaxprEqnRecipe) else []


def toposort(out_nodes: List[Any], parents: Callable[[Any], List[Any]]):
    if not out_nodes: return []
    out_nodes = remove_duplicates(out_nodes)

    child_counts = {}
    stack = list(out_nodes)
    while stack:
        node = stack.pop()
        if id(node) in child_counts:
            child_counts[id(node)] += 1
        else:
            child_counts[id(node)] = 1
            stack.extend(parents(node))
    for node in out_nodes:
        child_counts[id(node)] -= 1

    sorted_nodes = []
    childless_nodes = [node for node in out_nodes if not child_counts[id(node)]]
    while childless_nodes:
        node = childless_nodes.pop()
        sorted_nodes.append(node)
        for parent in parents(node):
            if child_counts[id(parent)] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[id(parent)] -= 1

    sorted_nodes = sorted_nodes[::-1]
    check_toposort(sorted_nodes, parents)

    return sorted_nodes


def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if id(x) not in seen and not seen.add(id(x))]


def check_toposort(nodes: List[Any], parents: Callable[[Any], List[Any]]):
    seen = set()
    for node in nodes:
        assert all(id(parent) in seen for parent in parents(node))
        seen.add(id(node))


# --- Linearize API ---

def partial_eval(fn: Callable, pvals_in: List[PartialVal]) -> Tuple[Jaxpr, List[PartialVal], List[Any]]:
    with new_main_trace(PartialEvalTrace) as main_trace:
        trace = PartialEvalTrace(main_trace)
        tracers_in = [trace.new_arg(pval) for pval in pvals_in]
        fn_outs = fn(*tracers_in)
        if not isinstance(fn_outs, (list, tuple)):
            fn_outs = (fn_outs,)
        tracers_out = [full_raise(trace, fn_out) for fn_out in fn_outs]
        pvals_out = [t.pval for t in tracers_out]
        unknown_tracers_in = list(filter(lambda t: t.pval.is_unknown, tracers_in))
        unknown_tracers_out = list(filter(lambda t: t.pval.is_unknown, tracers_out))
        jaxpr, consts = tracers_to_jaxpr(unknown_tracers_in, unknown_tracers_out)
    return jaxpr, pvals_out, consts


def linearize(fn, *primals_in):
    pvals_in = (
        [PartialVal.known(x) for x in primals_in] +
        [PartialVal.unknown(raise_val_to_shaped(x)) for x in primals_in]
    )

    def fn_jvp(*inputs):
        n = len(inputs) // 2
        primals_in, tangents_in = inputs[:n], inputs[n:]
        primals_out, tangents_out = jvp(fn, *primals_in, *tangents_in)
        if isinstance(primals_out, (list, tuple)):
            return [*primals_out, *tangents_out]
        else:
            return [primals_out, tangents_out]

    jaxpr, pvals_out, consts = partial_eval(fn_jvp, pvals_in)
    primal_pvals = pvals_out[:len(pvals_out)//2]
    assert all(pval.is_known for pval in primal_pvals)

    primals_out = [pval.const for pval in primal_pvals]
    if len(primals_out) == 1:
        primals_out = primals_out[0]
    linear_fn = lambda *tangents: eval_jaxpr(jaxpr, [*consts, *tangents])

    return primals_out, linear_fn
