from pangolin.ir import *
from engine import VmapEngine

engine = VmapEngine()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Row sweep — vary row index, fix column
# Expects: one vmap batching all adds along axis 0 of `a`
# ─────────────────────────────────────────────────────────────────────────────
def test_row_sweep():
    print("\n=== TEST 1: Row sweep (vary row, fix col) ===")
    a  = RV(Constant([[1,2,3],[4,5,6],[7,8,9]]))
    i0 = RV(Constant(0))
    i1 = RV(Constant(1))
    i2 = RV(Constant(2))
    col = RV(Constant(0))   # fixed column

    # x[0,0], x[1,0], x[2,0]  — same column, every row
    elems = [RV(Index(), a, RV(Constant(r)), col) for r in range(3)]
    # Add each element to itself → should batch into a single vmap
    adds  = [RV(Add(), e, e) for e in elems]
    engine.run_vmap([a, col, i0, i1, i2] + elems + adds)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Column sweep — fix row, vary column
# Expects: one vmap batching adds along axis 1 of `a`
# ─────────────────────────────────────────────────────────────────────────────
def test_col_sweep():
    print("\n=== TEST 2: Column sweep (fix row, vary col) ===")
    a   = RV(Constant([[10,20,30],[40,50,60]]))
    row = RV(Constant(0))

    elems = [RV(Index(), a, row, RV(Constant(c))) for c in range(3)]
    muls  = [RV(Mul(), e, e) for e in elems]
    engine.run_vmap([a, row] + elems + muls)

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
    engine.run_vmap([a] + elems + adds)

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
    engine.run_vmap([a, b] + ea + eb + adds)

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
    engine.run_vmap([a, b, b0] + ea + adds)

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
    engine.run_vmap([a] + ea + adds + muls)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Diagonal of a square matrix
# a[i,i] for i in range(N) — both indices vary together (same axis)
# ─────────────────────────────────────────────────────────────────────────────
def test_diagonal():
    print("\n=== TEST 7: Diagonal elements a[i,i] ===")
    a = RV(Constant([[1,2,3],[4,5,6],[7,8,9]]))

    diag = [RV(Index(), a, RV(Constant(i)), RV(Constant(i))) for i in range(3)]
    adds = [RV(Add(), d, d) for d in diag]
    engine.run_vmap([a] + diag + adds)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Large 1-D sweep — stress test greedy set cover
# ─────────────────────────────────────────────────────────────────────────────
def test_large_sweep():
    print("\n=== TEST 8: Large 1-D sweep (N=20) ===")
    N = 20
    a    = RV(Constant(list(range(N))))
    ea   = [RV(Index(), a, RV(Constant(i))) for i in range(N)]
    adds = [RV(Add(), e, e) for e in ea]
    engine.run_vmap([a] + ea + adds)

# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # test_row_sweep()
    # test_col_sweep()
    # test_full_grid()
    test_two_arrays_paired()
    test_broadcast_second_arg()
    test_mixed_ops_same_elems()
    test_diagonal()
    test_large_sweep()