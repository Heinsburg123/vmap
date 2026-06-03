from pangolin.ir import *
from pangolin import interface as pi
from engine import VmapEngine
from jags_pangolin.engine import Sample_prob

sample = Sample_prob().sample
engine = VmapEngine()


def test_autobatch_normal_shared_params():
    m = pi.constant(0)
    s = pi.constant(1)

    xs = [pi.normal(m, s) for _ in range(5)]

    d = engine.run_to_fixpoint([m,s]+xs)

    new_xs = [d[x] for x in xs]
    # graph_upstream(new_xs)
    for n, new_x in enumerate(new_xs):
        assert new_x.op == Index()
        batched, idx = new_x.parents
        assert batched.op == VMap(Normal(), (None, None), 5)
        print(batched.parents[0], m)
        assert rv_equal(batched.parents[0], m)
        assert rv_equal(batched.parents[1], s)

    batched_node = new_xs[0].parents[0]
    for new_x in new_xs:
        assert new_x.parents[0] is batched_node

def test_autobatch_normal_different_params():
    xs = [pi.normal(0,1) for _ in range(5)]
    d = engine.run_to_fixpoint(xs)
    new_xs = [d[x] for x in xs]
    graph_upstream(new_xs)
    for n, new_x in enumerate(new_xs):
        assert new_x.op == Index()
        batched, idx = new_x.parents
        assert batched.op == VMap(Normal(), (0, 0))
        # assert batched.parents[0] == Constant(0)
        # assert rv_equal(batched.parents[1], Constant(1))

    batched_node = new_xs[0].parents[0]
    for new_x in new_xs:
        assert new_x.parents[0] is batched_node
    graph_upstream([d[new_xs[i]] for i in range(5)])

def test_switch_order():
    xs = [pi.constant(i) for i in range(5)]
    ys = [x + 3 for x in [xs[3], xs[0], xs[4], xs[1], xs[2]]]
    d = engine.run_to_fixpoint(xs + ys)
    graph_upstream([d[ys[i]] for i in range(5)])

def test_switch_order_2():
    xs = pi.constant([0,1,2,3,4])
    ys = pi.constant([0,1,2,3,4])
    zs = [x + y for x, y in zip([xs[3], xs[0], xs[4], xs[1], xs[2]],
                                [ys[1], ys[0], ys[2], ys[4], ys[3]])]
    ws = [x + y for x, y in zip([xs[2], xs[0], xs[4], xs[1], xs[3]],
                                [ys[4], ys[0], ys[2], ys[1], ys[3]])]
    # graph_upstream([zs[i] for i in range(5)])
    d = engine.run_to_fixpoint([xs, ys]+zs+ws)
    print_upstream(list(d.items()))
    vmap = d[zs[0]].parents[0]
    for i in range(5):
        assert d[zs[i]].parents[0] is vmap
        assert d[ws[i]].parents[0] is vmap
    
    assert vmap.op == VMap(Add(), [0, 0])
    for i in range(5):
        assert d[zs[i]].parents[0].parents[0].parents[0] == xs
        assert d[zs[i]].parents[0].parents[1].parents[0] == ys
    assert [d[zs[i]].parents[0].parents[0].parents[1].op.value[d[zs[i]].parents[1].op.value.item()]for i in range(5)] == [3, 0, 4, 1, 2]
    assert [d[zs[i]].parents[0].parents[1].parents[1].op.value[d[zs[i]].parents[1].op.value.item()] for i in range(5)] == [1, 0, 2, 4, 3]
    assert [d[ws[i]].parents[0].parents[0].parents[1].op.value[d[ws[i]].parents[1].op.value.item()] for i in range(5)] == [2, 0, 4, 1, 3]
    assert [d[ws[i]].parents[0].parents[1].parents[1].op.value[d[ws[i]].parents[1].op.value.item()] for i in range(5)] == [4, 0, 2, 1, 3]
    #Problem of vectorizing over constants.

def test_same_parent():
    a = pi.constant([0,1,2,3,4])
    x = [pi.normal(ai,bi) for ai, bi in zip([a[0], a[1], a[2], a[3], a[4]], [a[0], a[1], a[2], a[3], a[4]])]
    d = engine.run_to_fixpoint([a] + x)
    graph_upstream([d[xi] for xi in x])


