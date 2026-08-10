from __future__ import annotations

import time
import numpy as np
import jax
from jax import numpy as jnp

from pangolin import blackjax as pblackjax
from pangolin import jax_backend
from pangolin.dag import upstream_nodes
from pangolin import interface as pi

from engine3 import VmapEngine


_orig_randint = np.random.randint


def _randint_int64_default(low, high=None, size=None, dtype=None):
    if dtype is None:
        dtype = np.int64
    return _orig_randint(low, high, size=size, dtype=dtype)


np.random.randint = _randint_int64_default

RNG = np.random.default_rng(0)

def index_all(arr_rv, n):
    return [arr_rv[i] for i in range(n)]

def model_1_random_slopes(n_groups=10, n_per_group=15):
    mu_a, tau_a = pi.constant(0.0), pi.constant(1.0)
    mu_b, tau_b = pi.constant(0.5), pi.constant(0.3)
    sigma = pi.constant(0.2)
    alphas = [pi.normal(mu_a, tau_a) for _ in range(n_groups)]
    betas = [pi.normal(mu_b, tau_b) for _ in range(n_groups)]

    ys, extra = [], []
    for g in range(n_groups):
        x_arr = pi.constant(RNG.normal(size=n_per_group))
        ex = [x_arr[i] for i in range(n_per_group)]
        for i in range(n_per_group):
            m =  alphas[g] +  betas[g]*ex[i]
            ys.append(pi.normal(m, sigma))

    return alphas + betas + ys


def model_1_random_slopes_vmapped(n_groups=10, n_per_group=15):
    mu_a, tau_a = pi.constant(0.0), pi.constant(1.0)
    mu_b, tau_b = pi.constant(0.5), pi.constant(0.3)
    sigma = pi.constant(0.2)
    alphas = pi.vmap(pi.normal, in_axes=None, axis_size=n_groups)(mu_a, tau_a)
    betas = pi.vmap(pi.normal, in_axes=None, axis_size=n_groups)(mu_b, tau_b)

    x_vals = RNG.normal(size=(n_groups, n_per_group))
    x_arr = pi.constant(x_vals)

    def fun(a, b, x):
        return pi.normal(a + b * x, sigma)

    ys = pi.vmap(pi.vmap(fun, [None, None, 0]), [0, 0, 0])(alphas, betas, x_arr)

    return [alphas, betas, ys]


def model_2_irt_2pl(n_persons=25, n_items=12):
    thetas = [pi.normal(0.0, 1.0) for _ in range(n_persons)]
    a_latents = [pi.normal(0.0, 0.3) for _ in range(n_items)]
    a_items = [pi.exp(a_lat) for a_lat in a_latents]
    b_items = [pi.normal(0.0, 1.0) for _ in range(n_items)]

    true_theta = RNG.normal(size=n_persons)
    true_a = np.exp(RNG.normal(scale=0.3, size=n_items))
    true_b = RNG.normal(size=n_items)

    ys, y_obs = [], []
    for p in range(n_persons):
        for j in range(n_items):
            logit = a_items[j] * (thetas[p] - b_items[j])
            ys.append(pi.bernoulli_logit(logit))
            true_logit = true_a[j] * (true_theta[p] - true_b[j])
            prob = 1.0 / (1.0 + np.exp(-true_logit))
            y_obs.append(int(RNG.binomial(1, prob)))

    targets = thetas + a_items + b_items
    return targets, ys, y_obs


def model_2_irt_2pl_vmapped(n_persons=25, n_items=12):
    thetas = pi.vmap(pi.normal, in_axes=None, axis_size=n_persons)(0.0, 1.0)
    a_latents = pi.vmap(pi.normal, in_axes=None, axis_size=n_items)(0.0, 0.3)
    a_items = pi.vmap(pi.exp)(a_latents)
    b_items = pi.vmap(pi.normal, in_axes=None, axis_size=n_items)(0.0, 1.0)

    true_theta = RNG.normal(size=n_persons)
    true_a = np.exp(RNG.normal(scale=0.3, size=n_items))
    true_b = RNG.normal(size=n_items)

    def fun(theta, a, b):
        logit = a * (theta - b)
        return pi.bernoulli_logit(logit)

    ys = pi.vmap(pi.vmap(fun, [None, 0, 0]), [0, None, None])(thetas, a_items, b_items)

    true_logit = true_a[None, :] * (true_theta[:, None] - true_b[None, :])
    prob = 1.0 / (1.0 + np.exp(-true_logit))
    y_obs = RNG.binomial(1, prob).astype(int)

    targets = [thetas, a_items, b_items]
    return targets, [ys], [y_obs]


def model_3_bayesian_matrix_factorization(n_users=12, n_items=10, d=3):

    U = [[pi.normal(0.0, 1.0) for _ in range(d)] for _ in range(n_users)]
    V = [[pi.normal(0.0, 1.0) for _ in range(d)] for _ in range(n_items)]
    sigma = pi.constant(0.3)

    ratings = []
    for i in range(n_users):
        for j in range(n_items):
            dot = U[i][0] * V[j][0]
            for k in range(1, d):
                dot = dot + U[i][k] * V[j][k]
            ratings.append(pi.normal(dot, sigma))

    flat_U = [u for row in U for u in row]
    flat_V = [v for row in V for v in row]
    targets = flat_U + flat_V + ratings
    return targets


def model_3_bayesian_matrix_factorization_vmapped(n_users=12, n_items=10, d=3):
    U = pi.vmap(pi.vmap(pi.normal, in_axes=None, axis_size=d), in_axes=None, axis_size=n_users)(0.0, 1.0)
    V = pi.vmap(pi.vmap(pi.normal, in_axes=None, axis_size=d), in_axes=None, axis_size=n_items)(0.0, 1.0)
    sigma = pi.constant(0.3)

    def fun(u, v):
        dot = pi.sum(u * v, 0)
        return pi.normal(dot, sigma)

    ratings = pi.vmap(pi.vmap(fun, [None, 0]), [0, None])(U, V)
    return [U, V, ratings]


def model_4_fir_time_series(T=60, K=4):
    x_vals = RNG.normal(size=T)
    x_arr = pi.constant(x_vals)
    ex = index_all(x_arr, T)
    weights = [pi.normal(0.0, 1.0) for _ in range(K)]
    sigma = pi.constant(0.2)

    ys = []
    for t in range(K - 1, T):
        m = weights[0] * ex[t]
        for k in range(1, K):
            m = m + weights[k] * ex[t - k]
        ys.append(pi.normal(m, sigma))

    targets = weights + ys
    return targets


def model_4_fir_time_series_vmapped(T=60, K=4):
    x_vals = RNG.normal(size=T)
    weights = pi.vmap(pi.normal, in_axes=None, axis_size=K)(0.0, 1.0)
    sigma = pi.constant(0.2)

    windows = np.stack([x_vals[t - K + 1:t + 1][::-1] for t in range(K - 1, T)])
    win_arr = pi.constant(windows)

    def fun(weights, row):
        m = pi.sum(weights * row, 0)
        return pi.normal(m, sigma)

    ys = pi.vmap(fun, [None, 0])(weights, win_arr)
    return [weights, ys]


def model_5_poisson_claims_with_exposure(n=150):
    x_vals = RNG.normal(size=n) * 0.2
    exposure_vals = RNG.uniform(0.5, 2.0, size=n)
    x_arr, exp_arr = pi.constant(x_vals), pi.constant(exposure_vals)
    beta0, beta1 = pi.constant(-1.0), pi.constant(0.5)
    ex, eexp = index_all(x_arr, n), index_all(exp_arr, n)

    rates, ys, y_obs = [], [], []
    for i in range(n):
        lin = beta0 + beta1 * ex[i]
        rate = eexp[i] * pi.exp(lin)
        rates.append(rate)
        ys.append(pi.poisson(rate))
        true_rate = exposure_vals[i] * np.exp(-1.0 + 0.5 * x_vals[i])
        y_obs.append(int(RNG.poisson(true_rate)))

    return rates, ys, y_obs


def model_5_poisson_claims_with_exposure_vmapped(n=150):
    x_vals = RNG.normal(size=n) * 0.2
    exposure_vals = RNG.uniform(0.5, 2.0, size=n)
    x_arr, exp_arr = pi.constant(x_vals), pi.constant(exposure_vals)
    beta0, beta1 = pi.constant(-1.0), pi.constant(0.5)

    def fun(xi, expi):
        lin = beta0 + beta1 * xi
        return expi * pi.exp(lin)

    rates = pi.vmap(fun)(x_arr, exp_arr)
    ys = pi.vmap(pi.poisson)(rates)

    true_rate = exposure_vals * np.exp(-1.0 + 0.5 * x_vals)
    y_obs = RNG.poisson(true_rate).astype(int)

    return [rates], [ys], [y_obs]


def model_6_kernel_weighted_regression(n_anchors=8, n_query=20):
    anchor_x = RNG.normal(size=n_anchors)
    anchor_y_vals = RNG.normal(size=n_anchors)
    query_x = RNG.normal(size=n_query)

    ax_arr, ay_arr, qx_arr = pi.constant(anchor_x), pi.constant(anchor_y_vals), pi.constant(query_x)
    eax, eay, eqx = index_all(ax_arr, n_anchors), index_all(ay_arr, n_anchors), index_all(qx_arr, n_query)
    sigma = pi.constant(0.3)

    means, ys = [], []
    for q in range(n_query):
        ks = []
        for a in range(n_anchors):
            diff = eqx[q] - eax[a]
            sqdist = diff * diff
            k = pi.exp(sqdist * -0.5)
            ks.append(k)
        num = ks[0] * eay[0]
        den = ks[0]
        for a in range(1, n_anchors):
            num = num + ks[a] * eay[a]
            den = den + ks[a]
        mean = num / den
        means.append(mean)
        ys.append(pi.normal(mean, sigma))

    targets = means + ys
    return targets


def model_6_kernel_weighted_regression_vmapped(n_anchors=8, n_query=20):
    anchor_x = RNG.normal(size=n_anchors)
    anchor_y_vals = RNG.normal(size=n_anchors)
    query_x = RNG.normal(size=n_query)

    ax_arr, ay_arr, qx_arr = pi.constant(anchor_x), pi.constant(anchor_y_vals), pi.constant(query_x)
    sigma = pi.constant(0.3)

    def kern_fun(ax, qx):
        diff = qx - ax
        return pi.exp(diff * diff * -0.5)

    ks = pi.vmap(pi.vmap(kern_fun, [0, None]), [None, 0])(ax_arr, qx_arr)  # (n_query, n_anchors)

    def weighted_fun(k_row, ay_arr):
        num = pi.sum(k_row * ay_arr, 0)
        den = pi.sum(k_row, 0)
        return num / den

    means = pi.vmap(weighted_fun, [0, None])(ks, ay_arr)

    def fun(m):
        return pi.normal(m, sigma)

    ys = pi.vmap(fun)(means)
    return [means, ys]


def model_7_meta_analysis_ragged(n_studies=14):
    arm_sizes = [RNG.integers(2, 10) for _ in range(n_studies)]
    mu, tau, sigma = pi.constant(0.0), pi.constant(1.0), pi.constant(0.5)
    thetas = [pi.normal(mu, tau) for _ in range(n_studies)]

    ys = []
    for s_idx, n_arm in enumerate(arm_sizes):
        ys += [pi.normal(thetas[s_idx], sigma) for _ in range(int(n_arm))]

    return thetas + ys


def model_7_meta_analysis_ragged_vmapped(n_studies=14):
    arm_sizes = [int(RNG.integers(2, 10)) for _ in range(n_studies)]
    mu, tau, sigma = pi.constant(0.0), pi.constant(1.0), pi.constant(0.5)
    thetas = pi.vmap(pi.normal, in_axes=None, axis_size=n_studies)(mu, tau)

    study_idx = np.concatenate([np.full(n, s) for s, n in enumerate(arm_sizes)])
    idx_arr = pi.constant(study_idx)

    def fun(thetas, idx):
        theta_s = pi.index(thetas, idx)
        return pi.normal(theta_s, sigma)

    ys = pi.vmap(fun, [None, 0])(thetas, idx_arr)
    return [thetas, ys]


def model_8_survival_covariate_shape(n=100):
    x_vals = RNG.normal(size=n) * 0.3
    x_arr = pi.constant(x_vals)
    ex = index_all(x_arr, n)
    beta, rate0 = pi.constant(0.4), pi.constant(1.0)

    rates, ts = [], []
    for i in range(n):
        shape = pi.exp(beta * ex[i])
        r = pi.gamma(shape, rate0)
        rates.append(r)
        ts.append(pi.exponential(r))

    return rates + ts


def model_8_survival_covariate_shape_vmapped(n=100):
    x_vals = RNG.normal(size=n) * 0.3
    x_arr = pi.constant(x_vals)
    beta, rate0 = pi.constant(0.4), pi.constant(1.0)

    def fun(xi):
        shape = pi.exp(beta * xi)
        r = pi.gamma(shape, rate0)
        t = pi.exponential(r)
        return r, t

    rates, ts = pi.vmap(fun)(x_arr)
    return [rates, ts]


def model_9_crossed_effects_multiplicative(n_subj=12, n_item=9):
    mu_a, tau_a = pi.constant(1.0), pi.constant(0.3)
    mu_b, tau_b = pi.constant(1.0), pi.constant(0.3)
    sigma = pi.constant(0.2)
    alphas = [pi.normal(mu_a, tau_a) for _ in range(n_subj)]
    betas = [pi.normal(mu_b, tau_b) for _ in range(n_item)]

    ys = []
    for s in range(n_subj):
        for i in range(n_item):
            m = alphas[s] * betas[i]
            ys.append(pi.normal(m, sigma))

    return alphas + betas + ys


def model_9_crossed_effects_multiplicative_vmapped(n_subj=12, n_item=9):
    mu_a, tau_a = pi.constant(1.0), pi.constant(0.3)
    mu_b, tau_b = pi.constant(1.0), pi.constant(0.3)
    sigma = pi.constant(0.2)
    alphas = pi.vmap(pi.normal, in_axes=None, axis_size=n_subj)(mu_a, tau_a)
    betas = pi.vmap(pi.normal, in_axes=None, axis_size=n_item)(mu_b, tau_b)

    def fun(a, b):
        m = a * b
        return pi.normal(m, sigma)

    ys = pi.vmap(pi.vmap(fun, [None, 0]), [0, None])(alphas, betas)
    return [alphas, betas, ys]


def model_10_heteroskedastic_robust_regression(n=130):
    x_vals = RNG.normal(size=n)
    z_vals = RNG.uniform(0.1, 1.0, size=n)
    x_arr, z_arr = pi.constant(x_vals), pi.constant(z_vals)
    a, b, nu, gamma = pi.constant(1.0), pi.constant(0.0), pi.constant(4.0), pi.constant(0.5)
    ex, ez = index_all(x_arr, n), index_all(z_arr, n)

    ts = []
    for i in range(n):
        m = a * ex[i] + b
        sig = pi.exp(gamma * ez[i])
        ts.append(pi.student_t(nu, m, sig))

    return ts


def model_10_heteroskedastic_robust_regression_vmapped(n=130):
    x_vals = RNG.normal(size=n)
    z_vals = RNG.uniform(0.1, 1.0, size=n)
    x_arr, z_arr = pi.constant(x_vals), pi.constant(z_vals)
    a, b, nu, gamma = pi.constant(1.0), pi.constant(0.0), pi.constant(4.0), pi.constant(0.5)

    def fun(xi, zi):
        m = a * xi + b
        sig = pi.exp(gamma * zi)
        return pi.student_t(nu, m, sig)

    ts = pi.vmap(fun)(x_arr, z_arr)
    return [ts]


MODELS = [
    ("random_slopes_mixed_effects", model_1_random_slopes, model_1_random_slopes_vmapped),
    ("irt_2pl_psychometrics", model_2_irt_2pl, model_2_irt_2pl_vmapped),
    ("bayesian_matrix_factorization", model_3_bayesian_matrix_factorization, model_3_bayesian_matrix_factorization_vmapped),
    ("fir_time_series_filter", model_4_fir_time_series, model_4_fir_time_series_vmapped),
    ("poisson_claims_with_exposure", model_5_poisson_claims_with_exposure, model_5_poisson_claims_with_exposure_vmapped),
    ("kernel_weighted_regression", model_6_kernel_weighted_regression, model_6_kernel_weighted_regression_vmapped),
    ("meta_analysis_ragged_groups", model_7_meta_analysis_ragged, model_7_meta_analysis_ragged_vmapped),
    ("survival_covariate_shape", model_8_survival_covariate_shape, model_8_survival_covariate_shape_vmapped),
    ("crossed_effects_multiplicative", model_9_crossed_effects_multiplicative, model_9_crossed_effects_multiplicative_vmapped),
    ("heteroskedastic_robust_regression", model_10_heteroskedastic_robust_regression, model_10_heteroskedastic_robust_regression_vmapped),
]


def _normalize_model_output(raw):
    if isinstance(raw, tuple):
        targets, given_vars, given_vals = raw
        return targets, given_vars, given_vals
    return raw, [], []

# ─────────────────────────────────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────────────────────────────────

def time_sample(vars_, niter, given_vars=None, given_vals=None):

    given_vars = given_vars or []
    given_vals = given_vals or []

    def _raw_time(n):
        t0 = time.perf_counter()
        samples = pblackjax.sample(vars_, given_vars, given_vals, niter=n)
        jax.block_until_ready(samples)
        t1 = time.perf_counter()
        return t1 - t0

    _ = _raw_time(niter)
    _ = _raw_time(niter)
    t_n = _raw_time(niter)

    return t_n

def build_mapped_batch(vars_, given_vars=None, given_vals=None):
    given_vars = given_vars or []
    given_vals = given_vals or []
    given_map = {rv: val for rv, val in zip(given_vars, given_vals)}

    eng = VmapEngine()
    all_nodes = list(vars_) + list(given_vars)
    mapped_dict = eng.run_all_vmaps(all_nodes, given_map)
    mapped_targets = [mapped_dict[v] for v in vars_]
    if given_vars:
        mapped_given_vars, mapped_given_vals = _build_mapped_values(
            given_vars, given_vals, mapped_dict
        )
    else:
        mapped_given_vars, mapped_given_vals = [], []

    return mapped_targets, mapped_given_vars, mapped_given_vals, mapped_dict


def sample_batch(vars_, given_vars=None, given_vals=None, niter=1000):

    given_vars = given_vars or []
    given_vals = given_vals or []


    mapped_targets, mapped_given_vars, mapped_given_vals, mapped_dict = build_mapped_batch(
        vars_, given_vars, given_vals
    )

    samples = pblackjax.sample(mapped_targets, mapped_given_vars, mapped_given_vals, niter=niter)
    jax.block_until_ready(samples)


def time_sample_batch(vars_, niter, given_vars=None, given_vals=None):
    given_vars = given_vars or []
    given_vals = given_vals or []
    mapped_targets, mapped_given_vars, mapped_given_vals, _ = build_mapped_batch(
        vars_, given_vars, given_vals
    )

    def _raw_time(n):
        t0 = time.perf_counter()
        samples = sample_batch(mapped_targets, mapped_given_vars, mapped_given_vals, niter=n)
        t1 = time.perf_counter()
        return t1 - t0

    _ = _raw_time(niter)
    _ = _raw_time(niter)
    t_n = _raw_time(niter)

    return t_n


def time_vmap_overhead(vars_, given_vars=None, given_vals=None):
    given_vars = given_vars or []
    given_vals = given_vals or []
    given_map = {rv: val for rv, val in zip(given_vars, given_vals)}
    all_nodes = list(vars_) + list(given_vars)

    t0 = time.perf_counter()
    eng = VmapEngine()
    eng.run_all_vmaps(all_nodes, given_map)
    t1 = time.perf_counter()
    return t1 - t0


def _resolve_index_chain_to_source(node):
    chain = []
    cur = node
    while cur.op.name == "Index":
        base = cur.parents[0]
        pos_val = np.asarray(cur.parents[1].op.value)
        if pos_val.ndim != 0:
            return cur, tuple(chain)
        chain.append(int(pos_val))
        cur = base
    chain.reverse()
    return cur, tuple(chain)

def _get_random_upstream(vars_):
    upstream_vars = upstream_nodes(vars_)
    return [n for n in upstream_vars if n.op.random]

def _build_mapped_values(raw_random_vars, raw_values, mapped_dict):
    source_coords = {}
    for r, val in zip(raw_random_vars, raw_values):
        m = mapped_dict[r]
        source, coord = _resolve_index_chain_to_source(m)
        bucket = source_coords.setdefault(source, {})
        if coord in bucket:
            raise AssertionError(
                f"Engine mapped two different raw random vars to the same "
                f"position {coord} of {source} -- this indicates an "
                f"indexing/grouping bug in the engine, not in this check."
            )
        bucket[coord] = np.asarray(val)

    mapped_vars, mapped_vals = [], []
    for source, coord_map in source_coords.items():
        if next(iter(coord_map)) == ():
            mapped_vars.append(source)
            mapped_vals.append(next(iter(coord_map.values())))
            continue
        shape = tuple(source.shape)
        sample_val = next(iter(coord_map.values()))
        dtype = np.asarray(sample_val).dtype
        arr = np.zeros(shape, dtype=dtype)
        for coord, v in coord_map.items():
            arr[coord] = v

        mapped_vars.append(source)
        mapped_vals.append(arr)

    return mapped_vars, mapped_vals


def check_correctness(raw_targets, mapped_dict, given_vars=None, given_vals=None,
                       n_checks=10, rtol=1e-3, atol=1e-3, seed=0):

    given_vars = given_vars or []
    given_vals = given_vals or []

    raw_random_all = _get_random_upstream(list(raw_targets) + list(given_vars))
    latent_vars = [v for v in raw_random_all if v not in given_vars]
    given_dict = {gv: jnp.asarray(val) for gv, val in zip(given_vars, given_vals)}

    key = jax.random.PRNGKey(seed)
    for check_i in range(n_checks):
        key, subkey = jax.random.split(key)
        sampled = jax_backend.ancestor_sample_flat(latent_vars, subkey)
        values_by_var = dict(zip(latent_vars, sampled))
        values_by_var.update(given_dict)

        raw_vars_ordered = raw_random_all
        raw_vals_ordered = [values_by_var[v] for v in raw_vars_ordered]

        logp_raw = float(jax_backend.ancestor_log_prob_flat(raw_vars_ordered, raw_vals_ordered))

        mapped_vars, mapped_vals = _build_mapped_values(
            raw_vars_ordered, raw_vals_ordered, mapped_dict
        )
        logp_mapped = float(jax_backend.ancestor_log_prob_flat(mapped_vars, mapped_vals))

        if not np.isclose(logp_raw, logp_mapped, rtol=rtol, atol=atol):
            raise AssertionError(
                f"Joint log-density mismatch on check {check_i+1}/{n_checks}: "
                f"raw={logp_raw!r}, mapped={logp_mapped!r}, "
                f"diff={abs(logp_raw - logp_mapped):.6g}"
            )




NITERS = [1000]


def run_benchmark():
    results = []

    for name, builder, vmapped_builder in MODELS:
        print(f"\n{'='*70}\nMODEL: {name}\n{'='*70}")

        raw_targets, given_vars, given_vals = _normalize_model_output(builder())
        given_map = {rv: val for rv, val in zip(given_vars, given_vals)}
        check_eng = VmapEngine()
        mapped_dict = check_eng.run_all_vmaps(list(raw_targets) + list(given_vars), given_map)

        try:
            check_correctness(raw_targets, mapped_dict, given_vars, given_vals)
            print("  correctness: PASS (10/10 deterministic log-density checks)")
        except AssertionError as e:
            print(f"  correctness: FAIL -- {e}")
            raise

        vmap_targets, vmap_given_vars, vmap_given_vals = _normalize_model_output(vmapped_builder())

        for niter in NITERS:

            raw_steady = time_sample(raw_targets, niter, given_vars, given_vals)
            mapped_steady = time_sample_batch(raw_targets, niter, given_vars, given_vals)
            handvmap_steady = time_sample(vmap_targets, niter, vmap_given_vars, vmap_given_vals)

            speedup = raw_steady / mapped_steady if mapped_steady > 0 else float("inf")
            handvmap_speedup = raw_steady / handvmap_steady if handvmap_steady > 0 else float("inf")

            print(
                f"  niter={niter:>6}  "
                f"raw={raw_steady:.4f}s  "
                f"mapped(engine+sample)={mapped_steady:.4f}s  "
                f"hand_vmapped={handvmap_steady:.4f}s  "
                f"speedup={speedup:.2f}x  "
                f"handvmap_speedup={handvmap_speedup:.2f}x"
            )

            results.append({
                "model": name,
                "niter": niter,
                "raw_runtime_s": raw_steady,
                "mapped_runtime_s": mapped_steady,
                "handvmap_runtime_s": handvmap_steady,
                "speedup": speedup,
                "handvmap_speedup": handvmap_speedup,
            })

    return results


def print_summary(results):
    print(f"\n{'='*120}\nSUMMARY\n{'='*120}")
    header = (
        f"{'model':<28}{'niter':>8}{'raw(s)':>12}"
        f"{'mapped(s)':>16}{'hand_vmap(s)':>14}{'speedup':>10}{'hv_speedup':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<28}{r['niter']:>8}"
            f"{r['raw_runtime_s']:>12.4f}{r['mapped_runtime_s']:>16.4f}"
            f"{r['handvmap_runtime_s']:>14.4f}{r['speedup']:>10.2f}{r['handvmap_speedup']:>12.2f}"
        )

if __name__ == "__main__":
    results = run_benchmark()
    print_summary(results)