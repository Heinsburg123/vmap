from pangolin.ir import *
from engine import VmapEngine

engine = VmapEngine()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Full grid — vary BOTH row and col
# Expects: the engine groups by (op, parent) but cannot collapse both axes
#          simultaneously; greedy set cover picks the best batching
# ─────────────────────────────────────────────────────────────────────────────
def test_full_grid():
    print("\n=== TEST 3: Full grid (vary row AND col) ===")
    a = RV(Constant([[1,2,3],[4,5,6],[7,8,9]]))

    elems = [
        RV(Index(), a, RV(Constant(r)), RV(Constant(c)))
        for r in range(3) for c in range(3)
    ]
    adds = [RV(Add(), e, e) for e in elems]
    M = engine.run_to_fixpoint([a] + elems + adds)
    # for rv in [a] + elems + adds:
    #     if(rv in M):
    #         print(f"{rv}: in vmap with in_axes={M[rv]}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Two arrays, matching index pattern
# Each Add takes one element from `a` and the corresponding element from `b`
# Expects: a single vmap with in_axes=(0, 0) because BOTH parents vary together
# ─────────────────────────────────────────────────────────────────────────────
def test_two_arrays_paired():
    print("\n=== TEST 4: Two arrays, paired indices ===")
    a = RV(Constant([1, 2, 3, 4, 5]))
    b = RV(Constant([10,20,30,40,50]))

    ea = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    eb = [RV(Index(), b, RV(Constant(i))) for i in range(5)]
    adds = [RV(Add(), ea[i], eb[i]) for i in range(5)]
    engine.run_to_fixpoint([a, b] + ea + eb + adds)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Two arrays, one fixed / one swept
# a[i] + b[0] for i in range(N)  — only the first parent varies
# Expects: vmap with in_axes=(0, None)
# ─────────────────────────────────────────────────────────────────────────────
def test_broadcast_second_arg():
    print("\n=== TEST 5: Sweep a, broadcast b[0] ===")
    a   = RV(Constant([1, 2, 3, 4, 5]))
    b   = RV(Constant([100, 200, 300]))
    b0  = RV(Index(), b, RV(Constant(0)))   # fixed scalar from b

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    adds = [RV(Add(), ea[i], b0) for i in range(5)]
    engine.run_to_fixpoint([a, b, b0] + ea + adds)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Mixed ops on the same elements
# Both Add and Mul on a[i] — should produce TWO separate vmaps (one per op)
# ─────────────────────────────────────────────────────────────────────────────
def test_mixed_ops_same_elems():
    print("\n=== TEST 6: Mixed ops (Add and Mul) on same swept elements ===")
    a = RV(Constant([1, 2, 3, 4, 5]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    adds = [RV(Add(), e, e) for e in ea]
    muls = [RV(Mul(), e, e) for e in ea]
    engine.run_to_fixpoint([a] + ea + adds + muls)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Large 1-D sweep — stress test greedy set cover
# ─────────────────────────────────────────────────────────────────────────────
def test_large_sweep():
    print("\n=== TEST 8: Large 1-D sweep (N=20) ===")
    N = 20
    a    = RV(Constant(list(range(N))))
    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(N)]
    adds = [RV(Add(), e, e) for e in ea]
    engine.run_to_fixpoint([a] + ea + adds)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: Normal distribution — mu swept, sigma fixed
# Normal(mu[i], sigma_fixed) for i in range(N)
# Expects: vmap with in_axes=(0, None)
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_sweep_mu():
    print("\n=== TEST 9: Normal — sweep mu, fix sigma ===")
    mus   = RV(Constant([0.0, 1.0, 2.0, 3.0, 4.0]))
    sigma = RV(Constant(1.0))   # fixed scalar, not indexed

    emu   = [RV(Index(), mus, RV(Constant(i))) for i in range(5)]
    norms = [RV(Normal(), e, sigma) for e in emu]
    engine.run_to_fixpoint([mus, sigma] + emu + norms)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: Normal distribution — both mu and sigma swept together
# Normal(mu[i], sigma[i]) for i in range(N)
# Expects: vmap with in_axes=(0, 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_sweep_both():
    print("\n=== TEST 10: Normal — sweep both mu and sigma ===")
    mus    = RV(Constant([0.0, 1.0, 2.0, 3.0]))
    sigmas = RV(Constant([0.5, 1.0, 1.5, 2.0]))

    emu    = [RV(Index(), mus,    RV(Constant(i))) for i in range(4)]
    esigma = [RV(Index(), sigmas, RV(Constant(i))) for i in range(4)]
    norms  = [RV(Normal(), emu[i], esigma[i]) for i in range(4)]
    engine.run_to_fixpoint([mus, sigmas] + emu + esigma + norms)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: Beta distribution — sweep alpha, fix beta
# Beta(alpha[i], beta_fixed) for i in range(N)
# Expects: vmap with in_axes=(0, None)
# ─────────────────────────────────────────────────────────────────────────────
def test_beta_sweep_alpha():
    print("\n=== TEST 11: Beta — sweep alpha, fix beta ===")
    alphas   = RV(Constant([0.5, 1.0, 2.0, 5.0]))
    beta_val = RV(Constant(1.0))

    ealpha = [RV(Index(), alphas, RV(Constant(i))) for i in range(4)]
    betas  = [RV(Beta(), e, beta_val) for e in ealpha]
    engine.run_to_fixpoint([alphas, beta_val] + ealpha + betas)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: Gamma distribution — sweep both shape (alpha) and rate (beta)
# Gamma(alpha[i], beta[i]) for i in range(N)
# Expects: vmap with in_axes=(0, 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_gamma_sweep_both():
    print("\n=== TEST 12: Gamma — sweep alpha and beta together ===")
    alphas = RV(Constant([1.0, 2.0, 3.0, 4.0]))
    betas  = RV(Constant([0.5, 1.0, 1.5, 2.0]))

    ealpha = [RV(Index(), alphas, RV(Constant(i))) for i in range(4)]
    ebeta  = [RV(Index(), betas,  RV(Constant(i))) for i in range(4)]
    gammas = [RV(Gamma(), ealpha[i], ebeta[i]) for i in range(4)]
    engine.run_to_fixpoint([alphas, betas] + ealpha + ebeta + gammas)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 13: Exponential — sweep rate parameter
# Exponential(rate[i]) for i in range(N)
# Expects: single vmap with in_axes=(0,)
# ─────────────────────────────────────────────────────────────────────────────
def test_exponential_sweep():
    print("\n=== TEST 13: Exponential — sweep rate ===")
    rates = RV(Constant([0.5, 1.0, 2.0, 4.0, 8.0]))

    erate = [RV(Index(), rates, RV(Constant(i))) for i in range(5)]
    exps  = [RV(Exponential(), e) for e in erate]
    engine.run_to_fixpoint([rates] + erate + exps)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 14: Poisson — sweep lambda
# Poisson(lambda[i]) for i in range(N)
# Expects: single vmap with in_axes=(0,)
# ─────────────────────────────────────────────────────────────────────────────
def test_poisson_sweep():
    print("\n=== TEST 14: Poisson — sweep lambda ===")
    lambdas = RV(Constant([1.0, 2.0, 5.0, 10.0]))

    elam    = [RV(Index(), lambdas, RV(Constant(i))) for i in range(4)]
    poissons = [RV(Poisson(), e) for e in elam]
    engine.run_to_fixpoint([lambdas] + elam + poissons)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 15: Bernoulli — sweep probability
# Bernoulli(p[i]) for i in range(N)
# Expects: single vmap with in_axes=(0,)
# ─────────────────────────────────────────────────────────────────────────────
def test_bernoulli_sweep():
    print("\n=== TEST 15: Bernoulli — sweep p ===")
    probs = RV(Constant([0.1, 0.3, 0.5, 0.7, 0.9]))

    ep    = [RV(Index(), probs, RV(Constant(i))) for i in range(5)]
    berns = [RV(Bernoulli(), e) for e in ep]
    engine.run_to_fixpoint([probs] + ep + berns)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 16: Unary ops — Exp, Log, Sin swept over same array
# Each produces its own vmap (different op → different hash bucket)
# Expects: THREE separate vmaps, one per unary op
# ─────────────────────────────────────────────────────────────────────────────
def test_unary_ops_same_array():
    print("\n=== TEST 16: Unary ops (Exp, Log, Sin) on same swept array ===")
    a = RV(Constant([0.1, 0.5, 1.0, 2.0, 3.0]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    exps = [RV(Exp(), e) for e in ea]
    logs = [RV(Log(), e) for e in ea]
    sins = [RV(Sin(), e) for e in ea]
    engine.run_to_fixpoint([a] + ea + exps + logs + sins)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 17: StudentT — sweep all three params (nu, mu, sigma)
# StudentT(nu[i], mu[i], sigma[i]) for i in range(N)
# Expects: vmap with in_axes=(0, 0, 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_studentt_sweep_all():
    print("\n=== TEST 17: StudentT — sweep nu, mu, sigma ===")
    nus    = RV(Constant([1.0, 2.0, 5.0, 10.0]))
    mus    = RV(Constant([0.0, 0.0, 1.0,  2.0]))
    sigmas = RV(Constant([1.0, 1.0, 2.0,  3.0]))

    enu    = [RV(Index(), nus,    RV(Constant(i))) for i in range(4)]
    emu    = [RV(Index(), mus,    RV(Constant(i))) for i in range(4)]
    esigma = [RV(Index(), sigmas, RV(Constant(i))) for i in range(4)]
    ts     = [RV(StudentT(), enu[i], emu[i], esigma[i]) for i in range(4)]
    engine.run_to_fixpoint([nus, mus, sigmas] + enu + emu + esigma + ts)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 18: Mixed distribution ops on the same swept mu
# Normal(mu[i], 1) and Cauchy(mu[i], 1) — same parent sweep, different dists
# Expects: TWO vmaps (one Normal, one Cauchy), both with in_axes=(0, None)
# ─────────────────────────────────────────────────────────────────────────────
def test_mixed_dists_same_parent():
    print("\n=== TEST 18: Normal vs Cauchy on same swept mu ===")
    mus   = RV(Constant([0.0, 1.0, 2.0, 3.0]))
    scale = RV(Constant(1.0))

    emu    = [RV(Index(), mus, RV(Constant(i))) for i in range(4)]
    norms  = [RV(Normal(), e, scale) for e in emu]
    cauchs = [RV(Cauchy(),  e, scale) for e in emu]
    engine.run_to_fixpoint([mus, scale] + emu + norms + cauchs)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 19: Arithmetic diversity — Sub, Div, Pow on same swept pair
# Expects: THREE separate vmaps (one per op), each with in_axes=(0, 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_arithmetic_diversity():
    print("\n=== TEST 19: Sub, Div, Pow on swept paired arrays ===")
    a = RV(Constant([2.0, 3.0, 4.0, 5.0]))
    b = RV(Constant([1.0, 1.0, 2.0, 2.0]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(4)]
    eb   = [RV(Index(), b, RV(Constant(i))) for i in range(4)]
    subs = [RV(Sub(), ea[i], eb[i]) for i in range(4)]
    divs = [RV(Div(), ea[i], eb[i]) for i in range(4)]
    pows = [RV(Pow(), ea[i], eb[i]) for i in range(4)]
    engine.run_to_fixpoint([a, b] + ea + eb + subs + divs + pows)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 20: Chained unary — Exp then Log cancel; Sin then Cos independent
# Exp(a[i]) and Log(Exp(a[i])): second layer has a non-Index parent (the Exp RV)
# Expects: Exp vmaps normally; Log sees a non-Index parent → NotIndex path
# ─────────────────────────────────────────────────────────────────────────────
def test_chained_unary():
    print("\n=== TEST 20: Chained unary (Exp → Log, Sin → Cos) ===")
    a = RV(Constant([0.5, 1.0, 1.5, 2.0]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(4)]
    exps = [RV(Exp(), e)  for e in ea]       # layer 1
    logs = [RV(Log(), ex) for ex in exps]    # layer 2: parent is Exp RV, not Index
    sins = [RV(Sin(), e)  for e in ea]
    coss = [RV(Cos(), s)  for s in sins]
    engine.run_to_fixpoint([a] + ea + exps + logs + sins + coss)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 21: Normal with Exp-transformed sigma
# mu[i] ~ Normal(raw_mu[i], exp(raw_sigma[i]))
# Both parents of Normal are non-Index RVs → NotIndex path for both
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_with_transformed_params():
    print("\n=== TEST 21: Normal(mu[i], exp(raw_sigma[i])) ===")
    raw_mu    = RV(Constant([0.0, 1.0, 2.0, 3.0]))
    raw_sigma = RV(Constant([-1.0, 0.0, 0.5, 1.0]))

    emu    = [RV(Index(), raw_mu,    RV(Constant(i))) for i in range(4)]
    esigma = [RV(Index(), raw_sigma, RV(Constant(i))) for i in range(4)]

    # Transform sigma through Exp (non-Index parents for Normal)
    tsigma = [RV(Exp(), e) for e in esigma]
    norms  = [RV(Normal(), emu[i], tsigma[i]) for i in range(4)]
    engine.run_to_fixpoint([raw_mu, raw_sigma] + emu + esigma + tsigma + norms)

# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_full_grid()
    test_two_arrays_paired()
    test_broadcast_second_arg()
    test_mixed_ops_same_elems()
    test_large_sweep()
    test_normal_sweep_mu()
    test_normal_sweep_both()
    test_beta_sweep_alpha()
    test_gamma_sweep_both()
    test_exponential_sweep()
    test_poisson_sweep()
    test_bernoulli_sweep()
    test_unary_ops_same_array()
    test_studentt_sweep_all()
    test_mixed_dists_same_parent()
    test_arithmetic_diversity()
    test_chained_unary()
    test_normal_with_transformed_params()