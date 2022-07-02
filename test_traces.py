import pytest
import numpy as np
import autodidax as adx


def test_eval():
    def scalar(x, y):
        a = 2.0 * adx.sin(x)
        b = 3.0 * adx.cos(y)
        c = a + (-b)
        return c

    def multi_output(x, y):
        return x > y, x < y

    print(scalar(2.0, 1.5))
    print(multi_output(2.0, 1.5))


def test_jvp_flat():
    def f(x):
        z = 2.0 * adx.sin(x)
        y = -z + x
        return y
    x, x_dot = 3.0, 1.0
    y, y_dot = adx.jvp(f, x, x_dot)

    assert np.allclose(y, -2.0 * np.sin(3.0) + 3.0)
    assert np.allclose(y_dot, -2.0 * np.cos(3.0) + 1.0)
    print(y, y_dot)


def test_derivative():
    def deriv(f):
        df_dx = lambda x: adx.jvp(f, x, 1.0)[1]
        return df_dx

    df = deriv(adx.sin)
    assert np.allclose(df(3.0), np.cos(3.0))

    d2f = deriv(deriv(adx.sin))
    assert np.allclose(d2f(3.0), -np.sin(3.0))

    d3f = deriv(deriv(deriv(adx.sin)))
    assert np.allclose(d3f(3.0), -np.cos(3.0))


def test_derive_control_flow():
    def deriv(f):
        df_dx = lambda x: adx.jvp(f, (x,), (1.0,))[1]
        return df_dx

    g = lambda x: 2.0 * x if x > 0.0 else x
    df = deriv(g)

    assert np.allclose(df(3.0), np.asarray(2.0))
    assert np.allclose(df(-3.0), np.asarray(1.0))


def test_vmap():
    def scalar_increment(x):
        assert np.ndim(x) == 0
        return x + 1

    vector_increment = adx.vmap(scalar_increment, 0)
    v = np.arange(3)
    w = vector_increment(v)
    assert np.all(w == v + np.ones(3))


    def jacfwd(fn, x):
        pushfwd = lambda v: adx.jvp(fn, (x,), (v,))[1]
        v_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
        return adx.vmap(pushfwd, 0)(v_in)

    jf = jacfwd(lambda x: adx.sin(x), np.arange(3.0))
    assert np.allclose(jf, np.eye(3) * np.cos(np.arange(3.0)))
    print(jf)


def test_jaxpr():
    aval = adx.jaxpr.raise_val_to_shaped(3.0)
    jaxpr, *_ = adx.make_jaxpr(lambda x: 2.0 * x, aval)
    print(jaxpr)
    print(adx.jaxpr.typecheck_jaxpr(jaxpr))

    aval = adx.jaxpr.raise_val_to_shaped(np.ones([2, 3]))
    jaxpr, *_ = adx.make_jaxpr(lambda x: adx.reduce_sum(x, 0), aval)
    print(jaxpr)
    print(adx.jaxpr.typecheck_jaxpr(jaxpr))


def test_jit():
    @adx.jit
    def fn_f(x, y):
        print('Tracing!')
        return adx.sin(x) * adx.cos(y)

    z = fn_f(3.0, 4.0)  # Should print 'Tracing'
    print(z)
    z = fn_f(4.0, 5.0)  # Should print nothing because of compilation cache hit
    print(z)

    fn_g = adx.jit(lambda x: adx.reduce_sum(x, axis=0))
    fn_out = fn_g(np.array([1., 2., 3.]))
    print(fn_out)

    def fn_h(x):
        y = adx.sin(x) * 2.
        z = - y + x
        return z
    def derive(fn):
        df = lambda x: adx.jvp(fn, x, 1.0)[1]
        return df
    ddf = derive(derive(fn_h))
    print(ddf(3.0))
    print(adx.jit(ddf)(3.0))


def test_jit_all():
    @adx.jit
    def fn(x):
        print('Tracing...')
        y = adx.sin(x) * 2.0
        z = -y + x
        return z

    x, x_dot = 3.0, 1.0
    y, y_dot = adx.jvp(fn, x, x_dot)
    print(y, y_dot)

    z, z_dot = adx.jvp(fn, 2.0, 1.0)
    print(z, z_dot)

    yb = adx.vmap(fn, 0)(np.arange(3.0))
    print(yb)


def test_multi_output_jit():
    z, z_dot = adx.jvp(lambda x: (x+1.0, x+2.0), 3.0, 1.0)
    print(z, z_dot)

    fn = adx.jit(lambda x: (x+1, x+2))
    print(fn(3))

    g = adx.vmap(fn, 0)
    print(g(np.arange(3)))


def test_linearize():
    y, sin_lin = adx.linearize(adx.sin, 3.)
    print(y, adx.sin(3.))
    print(sin_lin(1.), adx.cos(3.))
