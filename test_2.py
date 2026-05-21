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

    for n, new_x in enumerate(new_xs):
        assert new_x.op == ir.Index()
        batched, idx = new_x.parents
        assert batched.op == ir.VMap(ir.Normal(), (None, None), 5)
        assert batched.parents[0] == m
        assert batched.parents[1] == s

    batched_node = new_xs[0].parents[0]
    for new_x in new_xs:
        assert new_x.parents[0] is batched_node

def test_autobatch_normal_different_params():
    xs = [pi.normal(0,1) for _ in range(5)]
    d = engine.run_to_fixpoint(xs)
    new_xs = [d[x] for x in xs]

    for n, new_x in enumerate(new_xs):
        assert new_x.op == ir.Index()
        batched, idx = new_x.parents
        assert batched.op == ir.VMap(ir.Normal(), (None, None), 5)
        assert batched.parents[0] == Constant(0)
        assert batched.parents[1] == Constant(1)

    batched_node = new_xs[0].parents[0]
    for new_x in new_xs:
        assert new_x.parents[0] is batched_node

def test_switch_order():
    xs = [pi.constant(i) for i in range(5)]
    ys = [x + 3 for x in [xs[3], xs[0], xs[4], xs[1], xs[2]]]
    
    xs_new = pi.constant([0,1,2,3,4])
    tmp = xs_new[3,0,4,1,2]
    ys_new = pi.vmap(pi.add,[0,None])(tmp, 3)
