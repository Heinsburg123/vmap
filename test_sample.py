import time
import numpy as np
import jax
from jax import numpy as jnp

from pangolin import blackjax as pblackjax
from pangolin import jax_backend
from pangolin import interface as pi
from engine3 import VmapEngine
from benchmark import _build_mapped_values

engine = VmapEngine()

_orig_randint = np.random.randint


def _randint_int64_default(low, high=None, size=None, dtype=None):
    if dtype is None:
        dtype = np.int64
    return _orig_randint(low, high, size=size, dtype=dtype)


np.random.randint = _randint_int64_default


def sample_batch(vars_, niter, given_vars=None, given_vals=None):

    given_vars = given_vars or []
    given_vals = given_vals or []
    given_map = {rv: val for rv, val in zip(given_vars, given_vals)}

    t0 = time.perf_counter()

    all_nodes = list(vars_) + list(given_vars)
    mapped_dict = engine.run_all_vmaps(all_nodes, given_map)
    mapped_targets = [mapped_dict[v] for v in vars_]
    if given_vars:
        mapped_given_vars, mapped_given_vals = _build_mapped_values(
            given_vars, given_vals, mapped_dict
        )
    else:
        mapped_given_vars, mapped_given_vals = [], []

    samples = pblackjax.sample(mapped_targets, mapped_given_vars, mapped_given_vals, niter=niter)
    jax.block_until_ready(samples)

    t1 = time.perf_counter()

    return samples, t1 - t0, mapped_dict