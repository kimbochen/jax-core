import itertools as it
from typing import *


class NodeType(NamedTuple):
    name: str
    to_iterable: Callable
    from_iterable: Callable

    def __repr__(self):
        return f'NodeType({self.name})'


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_tree_defs: Tuple['PyTreeDef']


class SentinelPyTreeDef:
    ''' A special None type of PyTreeDef. '''
    def __repr__(self):
        return 'SentinelPyTreeDef'


NODE_TYPES: Dict[Type, NodeType] = {
    tuple: NodeType(str(tuple), lambda t: (None, t), lambda _, xs: tuple(xs)),
    list: NodeType(str(list), lambda l: (None, l), lambda _, xs: list(xs)),
    dict: NodeType(str(dict), lambda d: list(zip(*sorted(d.items()))), lambda k, v: dict(zip(k, v)))
}
LEAF = SentinelPyTreeDef()


def tree_flatten(x: Any) -> Tuple[List[Any], PyTreeDef]:
    node_type = NODE_TYPES.get(type(x))
    if node_type is not None:
        node_metadata, children = node_type.to_iterable(x)
        children_flat, child_trees = zip(*map(tree_flatten, children)) if children else ([], [])
        flattened = it.chain.from_iterable(children_flat)
        tree_def = PyTreeDef(node_type, node_metadata, tuple(child_trees))
        return list(flattened), tree_def
    else:
        return [x], LEAF


def tree_unflatten(tree_def: PyTreeDef, xs: Union[Any, Iterator]) -> Any:
    xs = xs if isinstance(xs, (list, tuple)) else (xs,)
    return tree_unflatten_helper(tree_def, iter(xs))

def tree_unflatten_helper(tree_def: PyTreeDef, xs: Iterator) -> Any:
    if tree_def is LEAF:
        return next(xs)
    else:
        children = (tree_unflatten_helper(t, xs) for t in tree_def.child_tree_defs)
        return tree_def.node_type.from_iterable(tree_def.node_metadata, children)


class SentinelValue:
    pass
EMPTY = SentinelValue()


class Store:
    value = EMPTY

    def set_value(self, val):
        assert self.value is EMPTY
        self.value = val

    def __call__(self):
        return self.value


def flatten_fn(fn, in_tree):
    store = Store()

    def flat_fn(*flat_args):
        pytree_args = tree_unflatten(in_tree, flat_args)
        out = fn(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fn, store
