from pangolin.ir import *
from pangolin import interface as pi
from engine3 import VmapEngine
from jags_pangolin.engine import Sample_prob
from jax import numpy as jnp
import numpy as np
import sys 

sample = Sample_prob().sample
engine = VmapEngine()
np.set_printoptions(threshold=sys.maxsize, linewidth=1000)
jnp.set_printoptions(threshold=sys.maxsize, linewidth=1000)

def test_merge_constants():
    xs = [pi.constant(2*i) for i in range(5)]
    M = engine.run_all_vmaps(xs)
    output = RV(Constant([2*i for i in range(5)]))
    print_upstream([M[rv] for rv in xs])
    for i,x in enumerate(xs):
        assert M[x].op == Index()
        assert rv_equal(M[x].parents[0],output)
        assert rv_equal(M[x].parents[1], RV(Constant(i)))

def test_merge_constants_downstream():
    xs = [pi.constant(2*i) for i in range(5)]
    ys = [x + 1 for x in xs]
    M = engine.run_all_vmaps(ys)
    print_upstream([M[y] for y in ys])
    arr = [2*i for i in range(5)]
    arr += [1 for _ in range(5)]
    x_new = RV(Index(), RV(Constant(arr)), RV(Constant([0,1,2,3,4])))
    ones = RV(Index(), RV(Constant(arr)), RV(Constant([5,6,7,8,9])))
    for y in ys:
        assert M[y].op == Index()
        vmap = M[y].parents[0]
        assert vmap.op == VMap(Add(),in_axes =[0,0])
        assert rv_equal(vmap.parents[0], x_new)
        assert rv_equal(vmap.parents[1], ones)

def test_autobatch_normal_shared_params():
    m = pi.constant(0)
    s = pi.constant(1)

    xs = [pi.normal(m, s) for _ in range(5)]

    M = engine.run_all_vmaps(xs)
    print_upstream([M[x] for x in xs])
    m_new = RV(Index(), RV(Constant([0,1])), RV(Constant(0)))
    s_new = RV(Index(), RV(Constant([0,1])), RV(Constant(1)))
    for x in xs:
        assert M[x].op == Index()
        vmap = M[x].parents[0]
        assert vmap.op == VMap(Normal(), in_axes=[None,None], axis_size = 5)
        assert rv_equal(m_new, vmap.parents[0])
        assert rv_equal(s_new, vmap.parents[1])

def test_autobatch_normal_halfshared_params():
    m = pi.constant(0)
    xs = [pi.normal(m, 1) for _ in range (5)]
    M = engine.run_all_vmaps(xs)
    print_upstream([M[x] for x in xs])

def test_autobatch_normal_different_params():
    xs = [pi.normal(0,1) for _ in range(5)]
    M = engine.run_all_vmaps(xs)
    print_upstream([M[x] for x in xs])
    num_new = RV(Constant([0,1,0,1,0,1,0,1,0,1]))
    m_new = RV(Index(), num_new, RV(Constant([0,2,4,6,8])))
    s_new = RV(Index(), num_new, RV(Constant([1,3,5,7,9])))
    for x in xs:
        assert M[x].op == Index()
        vmap = M[x].parents[0]
        assert vmap.op == VMap(Normal(), in_axes=[0,0])
        assert rv_equal(m_new, vmap.parents[0])
        assert rv_equal(s_new, vmap.parents[1])


def test_switch_order():
    xs = [pi.constant(i) for i in range(5)]
    ys = [x + 3 for x in [xs[3], xs[0], xs[4], xs[1], xs[2]]]
    M = engine.run_all_vmaps(xs + ys)
    print_upstream([M[y] for y in ys])
    num_new = [i for i in range(5)]
    num_new += [3 for _ in range(5)]
    num_new = RV(Constant(num_new))
    x_new = RV(Index(), num_new, RV(Constant([0,1,2,3,4])))
    y_new = RV(Index(), num_new, RV(Constant([6,8,9,5,7])))
    for y, idd in zip(ys,[3,0,4,1,2]):
        assert M[y].op == Index()
        vmap = M[y].parents[0]
        assert vmap.op == VMap(Add(), in_axes=[0,0])
        assert rv_equal(x_new, vmap.parents[0])
        assert rv_equal(y_new, vmap.parents[1])
        assert M[y].parents[1].op.value == idd

def test_switch_order_2():
    xs = pi.constant([0,1,2,3,4])
    ys = pi.constant([0,1,2,3,4])
    zs = [x + y for x, y in zip([xs[3], xs[0], xs[4], xs[1], xs[2]],
                                 [ys[1], ys[0], ys[2], ys[4], ys[3]])]
    ws = [x + y for x, y in zip([xs[2], xs[0], xs[4], xs[1], xs[3]],
                                 [ys[4], ys[0], ys[2], ys[1], ys[3]])]

    expected_xs = {zs[0]:3, zs[1]:0, zs[2]:4, zs[3]:1, zs[4]:2,
                   ws[0]:2, ws[1]:0, ws[2]:4, ws[3]:1, ws[4]:3}
    expected_ys = {zs[0]:1, zs[1]:0, zs[2]:2, zs[3]:4, zs[4]:3,
                   ws[0]:4, ws[1]:0, ws[2]:2, ws[3]:1, ws[4]:3}

    M = engine.run_all_vmaps(zs + ws)
    for rv in zs + ws:
        assert M[rv].op == Index()
        vmap = M[rv].parents[0]
        assert vmap.op == VMap(Add(), in_axes=[0,0])
        pos = M[rv].parents[1].op.value   
        xs_array = vmap.parents[0].parents[2].op.value 
        ys_array = vmap.parents[1].parents[2].op.value
        assert xs_array[pos] == expected_xs[rv]
        assert ys_array[pos] == expected_ys[rv]
    

def test_same_parent():
    a = pi.constant([0,1,2,3,4])
    x = [pi.normal(ai,bi) for ai, bi in zip([a[0], a[1], a[2], a[3], a[4]], [a[3], a[4], a[2], a[1], a[1]])]
    expected_a = {x[0]:0, x[1]:1, x[2]:2, x[3]:3, x[4]:4}
    expected_b = {x[0]:3, x[1]:4, x[2]:2, x[3]:1, x[4]:1}
    M = engine.run_all_vmaps(x)
    print_upstream([M[rv] for rv in x])
    for rv in x:
        assert M[rv].op == Index()
        vmap = M[rv].parents[0]
        assert vmap.op == VMap(Normal(), in_axes = [0,0])
        pos = M[rv].parents[1].op.value
        a_array = vmap.parents[0].op.value 
        b_array = vmap.parents[1].parents[1].op.value 
        assert a_array[pos] == expected_a[rv]
        assert b_array[pos] == expected_b[rv]

def test_add_1d_vector():
    a = pi.constant([0,1,2])
    b = pi.constant([2,3,4])
    c = pi.constant([3,4,5])
    e = [a+b, b+c, a+c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])
    expected_a = {e[0]:0, e[1]:1, e[2]:0}
    expected_b = {e[0]:1, e[1]:2, e[2]:2}
    for rv in e:
        assert M[rv].op == Index()
        vmap = M[rv].parents[0]
        assert vmap.op == VMap(VMap(Add(), in_axes=[0,0], axis_size=3), in_axes = [0,0])
        pos = M[rv].parents[1].op.value
        a_array = vmap.parents[0].parents[1].op.value
        b_array = vmap.parents[1].parents[1].op.value
        assert a_array[pos] == expected_a[rv]
        assert b_array[pos] == expected_b[rv]


def test_matmul_1d_vector():
    a = pi.constant([0,1,2])
    b = pi.constant([2,3,4])
    c = pi.constant([3,4,5])
    e = [a@b, b@c, a@c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])
    expected_a = {e[0]:0, e[1]:1, e[2]:0}
    expected_b = {e[0]:1, e[1]:2, e[2]:2}
    for rv in e:
        assert M[rv].op == Index()
        vmap = M[rv].parents[0]
        assert vmap.op == VMap(Matmul(), in_axes = [0,0])
        pos = M[rv].parents[1].op.value
        a_array = vmap.parents[0].parents[1].op.value
        b_array = vmap.parents[1].parents[1].op.value
        assert a_array[pos] == expected_a[rv]
        assert b_array[pos] == expected_b[rv]

def test_add_2d_vector():
    a = pi.constant([[0,1,2], [3,4,5]])
    b = pi.constant([[2,3,4], [5,6,7]])
    c = pi.constant([[3,4,5], [6,7,8]])
    e = [a+b, b+c, a+c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])
    expected_a = {e[0]:0, e[1]:1, e[2]:0}
    expected_b = {e[0]:1, e[1]:2, e[2]:2}
    for rv in e:
        assert M[rv].op == Index()
        vmap = M[rv].parents[0]
        assert vmap.op == VMap(VMap(VMap(Add(), in_axes=[0,0], axis_size = 3), in_axes=[0,0], axis_size=2), in_axes = [0,0])
        pos = M[rv].parents[1].op.value
        a_array = vmap.parents[0].parents[1].op.value
        b_array = vmap.parents[1].parents[1].op.value
        assert a_array[pos] == expected_a[rv]
        assert b_array[pos] == expected_b[rv]

def test_matmul_2d_vector():
    a = pi.constant([[0,1,2], [3,4,5], [6,7,8]])
    b = pi.constant([[2,3,4], [5,6,7], [8,9,10]])
    c = pi.constant([[3,4,5], [6,7,8], [9,10,11]])
    e = [a@b, b@c, a@c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])
    expected_a = {e[0]:0, e[1]:1, e[2]:0}
    expected_b = {e[0]:1, e[1]:2, e[2]:2}
    for rv in e:
        assert M[rv].op == Index()
        vmap = M[rv].parents[0]
        assert vmap.op == VMap(Matmul(), in_axes = [0,0])
        pos = M[rv].parents[1].op.value
        a_array = vmap.parents[0].parents[1].op.value
        b_array = vmap.parents[1].parents[1].op.value
        assert a_array[pos] == expected_a[rv]
        assert b_array[pos] == expected_b[rv]

def test_inner_product():
    a = pi.constant(0)
    b = pi.constant(1)
    c = pi.constant(3)
    d = pi.constant(4)
    e = pi.constant(5)
    f = pi.constant(6)
    g = a*b + c*d + e*f
    M = engine.run_all_vmaps([g])
    print_upstream([g])
    rv = M[g]
    assert rv.op == Add()
    assert rv.parents[1].op == Index()
    vmap1 = rv.parents[1].parents[0]
    assert vmap1.op == VMap(Mul(), in_axes = [0,0])
    assert rv_equal(vmap1.parents[0], RV(Index(), RV(Constant([0,1,3,4,5,6])) , RV(Constant([0,2,4]))))
    assert rv_equal(vmap1.parents[1], RV(Index(), RV(Constant([0,1,3,4,5,6])) , RV(Constant([1,3,5]))))
    assert rv.parents[0].op == Add()
    assert rv.parents[0].parents[1].op == Index()
    vmap2 = rv.parents[0].parents[1].parents[0]
    assert rv_equal(vmap1, vmap2)
    assert rv.parents[0].parents[0].op == Index()
    vmap3 = rv.parents[0].parents[0].parents[0]
    assert rv_equal(vmap1, vmap3)

def test_matrix_vector_product():
    a = [[1,2], [5,6]]
    b = [pi.constant(1), pi.constant(2)]
    c = [pi.constant(ai)*bi for ai, bi in zip(a[0], b)]
    e1 = c[0]+c[1]
    d = [pi.constant(ai)*bi for ai, bi in zip(a[1], b)]
    e2 = d[0]+d[1]
    M = engine.run_all_vmaps([e1, e2])
    print_upstream([M[e1], M[e2]])
    rv1 = M[e1]
    assert rv1.op == Index()
    vmap_add1 = rv1.parents[0]
    assert vmap_add1.op == VMap(Add(), in_axes =[0,0])
    assert vmap_add1.parents[1].op == Index()
    vmap_mul1 = vmap_add1.parents[1].parents[0]
    assert rv_equal(vmap_add1.parents[1].parents[1], RV(Constant([1,3])))
    assert vmap_mul1.op == VMap(Mul(), in_axes =[0,0])
    num_new = RV(Constant([1,2,1,2,5,6]))
    assert rv_equal(vmap_mul1.parents[0], RV(Index(), num_new, RV(Constant([2,3,4,5]))))
    assert rv_equal(vmap_mul1.parents[1], RV(Index(), num_new, RV(Constant([0,1,0,1]))))
    rv2 = M[e2]
    assert rv2.op == Index()
    vmap_add2 = rv2.parents[0]
    assert rv_equal(vmap_add2, vmap_add1)
    

def test_matrix_vector_product_2():
    a = [pi.constant([1,2]), pi.constant([5,6])]
    b = [pi.constant(1), pi.constant(2)]
    
    c = [ai*bi for ai, bi in zip(a[0], b)]
    e1 = c[0]+c[1]
    
    d = [ai*bi for ai, bi in zip(a[1], b)]
    e2 = d[0]+d[1]
    
    M = engine.run_all_vmaps([e1, e2])    
    print_upstream([M[e1], M[e2]])
    rv1 = M[e1]
    assert rv1.op == Index()
    vmap_add1 = rv1.parents[0]
    assert vmap_add1.op == VMap(Add(), in_axes=[0,0])
    
    vmap_mul_0 = vmap_add1.parents[0].parents[0]
    vmap_mul_1 = vmap_add1.parents[1].parents[0]
    assert vmap_mul_0.op == VMap(VMap(Mul(), in_axes = [0, None]), in_axes=[1,0])
    assert vmap_mul_1.op == VMap(VMap(Mul(), in_axes = [0, None]), in_axes=[1,0])
    assert rv_equal(vmap_mul_0, vmap_mul_1)
    
    rv2 = M[e2]
    assert rv2.op == Index()
    vmap_add2 = rv2.parents[0]
    
    assert rv_equal(vmap_add2, vmap_add1)

def test_matrix_matrix_product():
    A = [[pi.constant(1), pi.constant(2)],
         [pi.constant(3), pi.constant(4)]]
    
    B = [[pi.constant(5), pi.constant(6)],
         [pi.constant(7), pi.constant(8)]]

    C00 = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    C01 = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    
    C10 = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    C11 = A[1][0]*B[0][1] + A[1][1]*B[1][1]

    C_flat = [C00, C01, C10, C11]
    M = engine.run_all_vmaps(C_flat)
    print_upstream([M[c] for c in C_flat])
    
    for c in C_flat:
        rv = M[c]
        assert rv.op == Index()
        vmap_add = rv.parents[0]
        assert vmap_add.op == VMap(Add(), in_axes=[0,0])
        assert vmap_add.parents[1].op == Index()
        vmap_mul = vmap_add.parents[1].parents[0]
        assert vmap_mul.op == VMap(Mul(), in_axes=[0,0])
    
    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[1]].parents[0])
    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[2]].parents[0])
    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[3]].parents[0])

def test_matrix_matrix_product_2():
    A = pi.constant([[1, 2],
                     [3, 4]])
    B = pi.constant([[5, 6],
                     [7, 8]])

    c00 = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    c01 = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    
    c10 = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    c11 = A[1,0]*B[0,1] + A[1,1]*B[1,1]

    C_flat = [c00, c01, c10, c11]
    M = engine.run_all_vmaps(C_flat)
    print_upstream([M[c] for c in C_flat])

    for c in C_flat:
        rv = M[c]
        assert rv.op == Index()
        
        vmap_add = rv.parents[0]
        assert vmap_add.op == VMap(Add(), in_axes=[0,0])
        
        idx_1 = vmap_add.parents[0]
        idx_2 = vmap_add.parents[1]
        assert idx_1.op == Index()
        assert idx_2.op == Index()
        
        vmap_mul_nested = idx_1.parents[0]
        assert rv_equal(vmap_mul_nested, idx_2.parents[0])
        
        assert vmap_mul_nested.op == VMap(VMap(Mul(), in_axes=[0,0]), in_axes=[1,0])

    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[1]].parents[0])

def test_matrix_matrix_product_3():
    A = pi.constant([[1, 2],
                     [3, 4]])
                     
    b_col0 = pi.constant([5, 7])
    b_col1 = pi.constant([6, 8])

    c00 = A[0,0]*b_col0[0] + A[0,1]*b_col0[1]
    c10 = A[1,0]*b_col0[0] + A[1,1]*b_col0[1]

    c01 = A[0,0]*b_col1[0] + A[0,1]*b_col1[1]
    c11 = A[1,0]*b_col1[0] + A[1,1]*b_col1[1]

    C_flat = [c00, c10, c01, c11]
    M = engine.run_all_vmaps(C_flat)
    print_upstream([M[c] for c in C_flat])
    for c in C_flat:
        rv = M[c]
        assert rv.op == Index()
        
        vmap_add = rv.parents[0]
        assert vmap_add.op == VMap(Add(), in_axes=[0,0])
        
        idx_1 = vmap_add.parents[0]
        idx_2 = vmap_add.parents[1]
        assert idx_1.op == Index()
        assert idx_2.op == Index()
        
        vmap_mul_nested = idx_1.parents[0]
        assert rv_equal(vmap_mul_nested, idx_2.parents[0])
        
        assert vmap_mul_nested.op == VMap(VMap(Mul(), in_axes=[0,0]), in_axes=[1,1])

    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[1]].parents[0])

def test_weird():
    x = pi.constant([[0,1,2], [3,4,5]])
    ys = [x[0,0:2]*7, x[1,0:2]*8]
    M = engine.run_all_vmaps(ys)
    print_upstream([M[y] for y in ys])
    for y in ys:
        rv = M[y]
        assert rv.op == Index()
        
        vmap_mul_nested = rv.parents[0]

        assert vmap_mul_nested.op == VMap(VMap(Mul(), in_axes=[0, None], axis_size=2), in_axes=[0, 0])
        
    assert rv_equal(M[ys[0]].parents[0], M[ys[1]].parents[0])

def test_bayesian_neural_network_stress():

    n_inputs = 20     
    hidden = 15        

    w1 = [pi.normal(0, 1) for _ in range(hidden)]        # hidden weights
    b1 = [pi.normal(0, 1) for _ in range(hidden)]        # hidden biases
    w2 = [pi.normal(0, 1) for _ in range(hidden)]        # output weights
    b2 = pi.normal(0, 1)                                  # output bias
    sigma = pi.constant(0.5)

    xs = [pi.constant(float(i)) for i in range(n_inputs)]

    ys = []
    for x in xs:
        hidden_acts = [pi.tanh(w1[j] * x + b1[j]) for j in range(hidden)]
        out = b2
        for j in range(hidden):
            out = out + w2[j] * hidden_acts[j]
        y = pi.normal(out, sigma)
        ys.append(y)

    M = engine.run_all_vmaps(ys)
    print_upstream([M[y] for y in ys])

def test_hierarchical_model_practical():
    n_groups = 8
    n_obs_per_group = 5

    mu_pop = pi.normal(0, 10)      
    tau = pi.constant(1.0)         
    sigma = pi.constant(0.5)   
    
    group_means = [pi.normal(mu_pop, tau) for _ in range(n_groups)]

    observations = []
    for g in range(n_groups):
        for i in range(n_obs_per_group):
            y = pi.normal(group_means[g], sigma)
            observations.append(y)

    M = engine.run_all_vmaps(observations)
    print_upstream([M[y] for y in observations])

test_hierarchical_model_practical()

