# src/utils/seed.py
import jax.random as jr
def make_seeds(seed: int, n: int = 3):
    return jr.split(jr.PRNGKey(seed), n)
