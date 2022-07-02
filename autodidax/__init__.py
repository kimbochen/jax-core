from autodidax.core import POP
from autodidax.auto_diff import jvp
from autodidax.batching import vmap
from autodidax.jaxpr import make_jaxpr
from autodidax.linearize import linearize
from autodidax.xla import jit


add = POP.add
mul = POP.mul
neg = POP.neg
sin = POP.sin
cos = POP.cos
greater = POP.greater
less = POP.less
transpose = POP.transpose
broadcast = POP.broadcast
reduce_sum = POP.reduce_sum
