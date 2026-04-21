from pangolin.ir import *
from engine import VmapEngine
from jags_pangolin.engine import Sample_prob

sample = Sample_prob().sample
engine = VmapEngine()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Full grid — vary BOTH row and col
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

    raw    = sample(adds)
    mapped = sample([M[add] for add in adds])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Two arrays, matching index pattern
# Expects: a single vmap with in_axes=(0, 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_two_arrays_paired():
    print("\n=== TEST 4: Two arrays, paired indices ===")
    a = RV(Constant([1, 2, 3, 4, 5]))
    b = RV(Constant([10,20,30,40,50]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    eb   = [RV(Index(), b, RV(Constant(i))) for i in range(5)]
    adds = [RV(Add(), ea[i], eb[i]) for i in range(5)]
    M    = engine.run_to_fixpoint([a, b] + ea + eb + adds)

    raw     = sample(adds)
    mapped  = sample([M[add] for add in adds])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Two arrays, one fixed / one swept
# Expects: vmap with in_axes=(0, None)
# ─────────────────────────────────────────────────────────────────────────────
def test_broadcast_second_arg():
    print("\n=== TEST 5: Sweep a, broadcast b[0] ===")
    a  = RV(Constant([1, 2, 3, 4, 5]))
    b  = RV(Constant([100, 200, 300]))
    b0 = RV(Index(), b, RV(Constant(0)))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    adds = [RV(Add(), ea[i], b0) for i in range(5)]
    M    = engine.run_to_fixpoint([a, b, b0] + ea + adds)

    raw    = sample(adds)
    mapped = sample([M[add] for add in adds])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Mixed ops on the same elements
# Expects: TWO separate vmaps (one Add, one Mul)
# ─────────────────────────────────────────────────────────────────────────────
def test_mixed_ops_same_elems():
    print("\n=== TEST 6: Mixed ops (Add and Mul) on same swept elements ===")
    a = RV(Constant([1, 2, 3, 4, 5]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    adds = [RV(Add(), e, e) for e in ea]
    muls = [RV(Mul(), e, e) for e in ea]
    M    = engine.run_to_fixpoint([a] + ea + adds + muls)

    raw_adds    = sample(adds)
    mapped_adds = sample([M[add] for add in adds])
    raw_muls    = sample(muls)
    mapped_muls = sample([M[mul] for mul in muls])

    for r, m in zip(raw_adds, mapped_adds):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Add mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"
    for r, m in zip(raw_muls, mapped_muls):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Mul mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Large 1-D sweep — stress test greedy set cover
# ─────────────────────────────────────────────────────────────────────────────
def test_large_sweep():
    print("\n=== TEST 8: Large 1-D sweep (N=20) ===")
    N    = 20
    a    = RV(Constant(list(range(N))))
    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(N)]
    adds = [RV(Add(), e, e) for e in ea]
    M    = engine.run_to_fixpoint([a] + ea + adds)

    raw    = sample(adds)
    mapped = sample([M[add] for add in adds])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: Normal distribution — mu swept, sigma fixed
# Expects: vmap with in_axes=(0, None)
# Stochastic: use looser tolerance on sample means
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_sweep_mu():
    print("\n=== TEST 9: Normal — sweep mu, fix sigma ===")
    mus   = RV(Constant([0.0, 1.0, 2.0, 3.0, 4.0]))
    sigma = RV(Constant(1.0))

    emu   = [RV(Index(), mus, RV(Constant(i))) for i in range(5)]
    norms = [RV(Normal(), e, sigma) for e in emu]
    M     = engine.run_to_fixpoint([mus, sigma] + emu + norms)

    raw    = sample(norms)
    mapped = sample([M[n] for n in norms])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"Normal mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: Normal distribution — both mu and sigma swept together
# Expects: vmap with in_axes=(0, 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_sweep_both():
    print("\n=== TEST 10: Normal — sweep both mu and sigma ===")
    mus    = RV(Constant([0.0, 1.0, 2.0, 3.0]))
    sigmas = RV(Constant([0.5, 1.0, 1.5, 2.0]))

    emu    = [RV(Index(), mus,    RV(Constant(i))) for i in range(4)]
    esigma = [RV(Index(), sigmas, RV(Constant(i))) for i in range(4)]
    norms  = [RV(Normal(), emu[i], esigma[i]) for i in range(4)]
    M      = engine.run_to_fixpoint([mus, sigmas] + emu + esigma + norms)

    raw    = sample(norms)
    mapped = sample([M[n] for n in norms])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"Normal mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: Beta distribution — sweep alpha, fix beta
# Expects: vmap with in_axes=(0, None)
# ─────────────────────────────────────────────────────────────────────────────
def test_beta_sweep_alpha():
    print("\n=== TEST 11: Beta — sweep alpha, fix beta ===")
    alphas   = RV(Constant([0.5, 1.0, 2.0, 5.0]))
    beta_val = RV(Constant(1.0))

    ealpha = [RV(Index(), alphas, RV(Constant(i))) for i in range(4)]
    betas  = [RV(Beta(), e, beta_val) for e in ealpha]
    M      = engine.run_to_fixpoint([alphas, beta_val] + ealpha + betas)

    raw    = sample(betas)
    mapped = sample([M[b] for b in betas])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"Beta mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: Gamma distribution — sweep both shape and rate
# Expects: vmap with in_axes=(0, 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_gamma_sweep_both():
    print("\n=== TEST 12: Gamma — sweep alpha and beta together ===")
    alphas = RV(Constant([1.0, 2.0, 3.0, 4.0]))
    betas  = RV(Constant([0.5, 1.0, 1.5, 2.0]))

    ealpha = [RV(Index(), alphas, RV(Constant(i))) for i in range(4)]
    ebeta  = [RV(Index(), betas,  RV(Constant(i))) for i in range(4)]
    gammas = [RV(Gamma(), ealpha[i], ebeta[i]) for i in range(4)]
    M      = engine.run_to_fixpoint([alphas, betas] + ealpha + ebeta + gammas)

    raw    = sample(gammas)
    mapped = sample([M[g] for g in gammas])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"Gamma mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 13: Exponential — sweep rate parameter
# Expects: single vmap with in_axes=(0,)
# ─────────────────────────────────────────────────────────────────────────────
def test_exponential_sweep():
    print("\n=== TEST 13: Exponential — sweep rate ===")
    rates = RV(Constant([0.5, 1.0, 2.0, 4.0, 8.0]))

    erate = [RV(Index(), rates, RV(Constant(i))) for i in range(5)]
    exps  = [RV(Exponential(), e) for e in erate]
    M     = engine.run_to_fixpoint([rates] + erate + exps)

    raw    = sample(exps)
    mapped = sample([M[e] for e in exps])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"Exponential mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 14: Poisson — sweep lambda
# Expects: single vmap with in_axes=(0,)
# ─────────────────────────────────────────────────────────────────────────────
def test_poisson_sweep():
    print("\n=== TEST 14: Poisson — sweep lambda ===")
    lambdas  = RV(Constant([1.0, 2.0, 5.0, 10.0]))

    elam     = [RV(Index(), lambdas, RV(Constant(i))) for i in range(4)]
    poissons = [RV(Poisson(), e) for e in elam]
    M        = engine.run_to_fixpoint([lambdas] + elam + poissons)

    raw    = sample(poissons)
    mapped = sample([M[p] for p in poissons])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=1.0), \
            f"Poisson mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 15: Bernoulli — sweep probability
# Expects: single vmap with in_axes=(0,)
# ─────────────────────────────────────────────────────────────────────────────
def test_bernoulli_sweep():
    print("\n=== TEST 15: Bernoulli — sweep p ===")
    probs = RV(Constant([0.1, 0.3, 0.5, 0.7, 0.9]))

    ep    = [RV(Index(), probs, RV(Constant(i))) for i in range(5)]
    berns = [RV(Bernoulli(), e) for e in ep]
    M     = engine.run_to_fixpoint([probs] + ep + berns)

    raw    = sample(berns)
    mapped = sample([M[b] for b in berns])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.2), \
            f"Bernoulli mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 16: Unary ops — Exp, Log, Sin swept over same array
# Expects: THREE separate vmaps, one per unary op
# ─────────────────────────────────────────────────────────────────────────────
def test_unary_ops_same_array():
    print("\n=== TEST 16: Unary ops (Exp, Log, Sin) on same swept array ===")
    a = RV(Constant([0.1, 0.5, 1.0, 2.0, 3.0]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(5)]
    exps = [RV(Exp(), e) for e in ea]
    logs = [RV(Log(), e) for e in ea]
    sins = [RV(Sin(), e) for e in ea]
    M    = engine.run_to_fixpoint([a] + ea + exps + logs + sins)

    raw_exps    = sample(exps)
    mapped_exps = sample([M[e] for e in exps])
    raw_logs    = sample(logs)
    mapped_logs = sample([M[l] for l in logs])
    raw_sins    = sample(sins)
    mapped_sins = sample([M[s] for s in sins])

    for r, m in zip(raw_exps, mapped_exps):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Exp mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"
    for r, m in zip(raw_logs, mapped_logs):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Log mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"
    for r, m in zip(raw_sins, mapped_sins):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Sin mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 17: StudentT — sweep all three params (nu, mu, sigma)
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
    M      = engine.run_to_fixpoint([nus, mus, sigmas] + enu + emu + esigma + ts)

    raw    = sample(ts)
    mapped = sample([M[t] for t in ts])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"StudentT mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 18: Mixed distribution ops on the same swept mu
# Expects: TWO vmaps (one Normal, one Cauchy), both with in_axes=(0, None)
# ─────────────────────────────────────────────────────────────────────────────
def test_mixed_dists_same_parent():
    print("\n=== TEST 18: Normal vs Cauchy on same swept mu ===")
    mus   = RV(Constant([0.0, 1.0, 2.0, 3.0]))
    scale = RV(Constant(1.0))

    emu    = [RV(Index(), mus, RV(Constant(i))) for i in range(4)]
    norms  = [RV(Normal(), e, scale) for e in emu]
    cauchs = [RV(Cauchy(),  e, scale) for e in emu]
    M      = engine.run_to_fixpoint([mus, scale] + emu + norms + cauchs)

    raw_norms    = sample(norms)
    mapped_norms = sample([M[n] for n in norms])
    raw_cauchs   = sample(cauchs)
    mapped_cauchs = sample([M[c] for c in cauchs])

    for r, m in zip(raw_norms, mapped_norms):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"Normal mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"
    for r, m in zip(raw_cauchs, mapped_cauchs):
        assert np.isclose(np.mean(r), np.mean(m), atol=1.0), \
            f"Cauchy mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

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
    M    = engine.run_to_fixpoint([a, b] + ea + eb + subs + divs + pows)

    raw_subs    = sample(subs)
    mapped_subs = sample([M[s] for s in subs])
    raw_divs    = sample(divs)
    mapped_divs = sample([M[d] for d in divs])
    raw_pows    = sample(pows)
    mapped_pows = sample([M[p] for p in pows])

    for r, m in zip(raw_subs, mapped_subs):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Sub mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"
    for r, m in zip(raw_divs, mapped_divs):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Div mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"
    for r, m in zip(raw_pows, mapped_pows):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Pow mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 20: Chained unary — Exp then Log cancel; Sin then Cos independent
# ─────────────────────────────────────────────────────────────────────────────
def test_chained_unary():
    print("\n=== TEST 20: Chained unary (Exp → Log, Sin → Cos) ===")
    a = RV(Constant([0.5, 1.0, 1.5, 2.0]))

    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(4)]
    exps = [RV(Exp(), e) for e in ea]
    logs = [RV(Log(), ex) for ex in [exps[3], exps[0], exps[1], exps[2]]]
    sins = [RV(Sin(), e) for e in ea]
    coss = [RV(Cos(), s) for s in sins]
    stuff = [RV(Add(), e, l) for (e, l) in zip(exps, logs, strict=True)]
    M = engine.run_to_fixpoint([a] + ea + exps + logs + sins + coss + stuff)

    raw_stuff    = sample(stuff)
    mapped_stuff = sample([M[s] for s in stuff])
    raw_coss     = sample(coss)
    mapped_coss  = sample([M[c] for c in coss])

    for r, m in zip(raw_stuff, mapped_stuff):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Add(Exp,Log) mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"
    for r, m in zip(raw_coss, mapped_coss):
        assert np.isclose(np.mean(r), np.mean(m), atol=1e-5), \
            f"Cos mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 21: Normal with Exp-transformed sigma
# Expects: NotIndex path for both Normal parents
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_with_transformed_params():
    print("\n=== TEST 21: Normal(mu[i], exp(raw_sigma[i])) ===")
    raw_mu    = RV(Constant([0.0, 1.0, 2.0, 3.0]))
    raw_sigma = RV(Constant([-1.0, 0.0, 0.5, 1.0]))

    emu    = [RV(Index(), raw_mu,    RV(Constant(i))) for i in range(4)]
    esigma = [RV(Index(), raw_sigma, RV(Constant(i))) for i in range(4)]
    tsigma = [RV(Exp(), e) for e in esigma]
    norms  = [RV(Normal(), emu[i], tsigma[i]) for i in range(4)]
    M      = engine.run_to_fixpoint([raw_mu, raw_sigma] + emu + esigma + tsigma + norms)

    raw    = sample(norms)
    mapped = sample([M[n] for n in norms])

    for r, m in zip(raw, mapped):
        assert np.isclose(np.mean(r), np.mean(m), atol=0.5), \
            f"Normal(transformed) mean mismatch: raw={np.mean(r)}, mapped={np.mean(m)}"

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