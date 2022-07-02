# JAX Core


## Interpreters

- Evaluate
- Jacobian-Vector Product (jvp)
- Vectorization (vmap)
- Dynamic tracing (make_jaxpr)
- Just-in-Time Compilation (jit)


## Evaluate

Example:
```python
y = 2.0 * adx.sin(3.0)
```

1. `adx.sin` intercepts the argument (3.0) and passes it to `bind`.
2. `bind` does the following:
  1. Selects a trace T based on a tracer's main trace (i.e. interpreter) levels.
  2. Wrap arguments into tracers of T.
  3. Apply T's interpretation rule of the primitive (in this case, `sin`), which outputs Q.
  4. Unwrap tracers Q into values.
