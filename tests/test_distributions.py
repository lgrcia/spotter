import jax
import jax.numpy as jnp

from spotter import distributions


def test_butterfly_jax():
    key = jax.random.PRNGKey(0)
    distributions.jax_butterfly(key)

    calc = jnp.array(distributions.jax_butterfly(key, n=20))
    assert calc.shape == (2, 20)


def test_uniform_jax():
    key = jax.random.PRNGKey(0)
    distributions.jax_uniform(key)

    calc = jnp.array(distributions.jax_uniform(key, n=20))
    assert calc.shape == (2, 20)


def test_butterfly():
    distributions.butterfly()

    calc = jnp.array(distributions.butterfly(n=20))
    assert calc.shape == (2, 20)


def test_uniform():
    distributions.uniform()

    calc = jnp.array(distributions.uniform(n=20))
    assert calc.shape == (2, 20)
