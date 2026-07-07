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

def test_merge_constants_downstream():
    xs = [pi.constant(2*i) for i in range(5)]
    ys = [x + 1 for x in xs]
    M = engine.run_all_vmaps(ys)
    print_upstream([M[y] for y in ys])
    x_new = pi.constant([0, 2, 4, 6, 8])
    y_new = pi.constant(1)
    vmap = M[ys[0]].parents[0]
    assert vmap.op == VMap(Add(),in_axes =[0,None])
    assert rv_equal(vmap.parents[0], x_new)
    assert rv_equal(vmap.parents[1], y_new)
    for i, y in enumerate(ys):
        assert M[y].op == Index()
        assert M[y].parents[0] == vmap
        assert M[y].parents[1].op == Constant(i)

def test_autobatch_normal_shared_params():
    m = pi.constant(0)
    s = pi.constant(1)

    xs = [pi.normal(m, s) for _ in range(5)]

    M = engine.run_all_vmaps(xs)
    print_upstream([M[x] for x in xs])

    vmap = M[xs[0]].parents[0]
    assert vmap.op == VMap(Normal(), in_axes=[None, None], axis_size=5)
    assert rv_equal(vmap.parents[0], pi.constant(0))
    assert rv_equal(vmap.parents[1], pi.constant(1))

    for i, x in enumerate(xs):
        assert M[x].op == Index()
        assert M[x].parents[0] == vmap
        assert M[x].parents[1].op == Constant(i)

# test_autobatch_normal_shared_params()

def test_autobatch_normal_halfshared_params():
    m = pi.constant(0)
    xs = [pi.normal(m, 1) for _ in range(5)]
    M = engine.run_all_vmaps(xs)
    print_upstream([M[x] for x in xs])

    vmap = M[xs[0]].parents[0]
    assert vmap.op == VMap(Normal(), in_axes=[None, None], axis_size=5)
    assert rv_equal(vmap.parents[0], pi.constant(0))
    assert rv_equal(vmap.parents[1], pi.constant(1))

    for i, x in enumerate(xs):
        assert M[x].op == Index()
        assert M[x].parents[0] == vmap
        assert M[x].parents[1].op == Constant(i)

# test_autobatch_normal_halfshared_params()

def test_autobatch_normal_different_params():
    xs = [pi.normal(0, 1) for _ in range(5)]
    M = engine.run_all_vmaps(xs)
    print_upstream([M[x] for x in xs])

    vmap = M[xs[0]].parents[0]
    assert vmap.op == VMap(Normal(), in_axes=[None, None], axis_size=5)
    assert rv_equal(pi.constant(0), vmap.parents[0])
    assert rv_equal(pi.constant(1), vmap.parents[1])

    for i, x in enumerate(xs):
        assert M[x].op == Index()
        assert M[x].parents[0] == vmap
        assert M[x].parents[1].op == Constant(i)

# test_autobatch_normal_different_params()

def test_switch_order():
    xs = [pi.constant(i) for i in range(5)]
    ys = [x + 3 for x in [xs[3], xs[0], xs[4], xs[1], xs[2]]]
    M = engine.run_all_vmaps(xs + ys)
    print_upstream([M[y] for y in ys])

    vmap = M[ys[0]].parents[0]
    assert vmap.op == VMap(Add(), in_axes=[0, None])
    assert rv_equal(pi.constant([3, 0, 4, 1, 2]), vmap.parents[0])
    assert rv_equal(pi.constant(3), vmap.parents[1])

    for i,y in enumerate(ys):
        assert M[y].op == Index()
        assert M[y].parents[0] == vmap
        assert M[y].parents[1].op == Constant(i)

def test_same_parent():

    a = pi.constant([0.0,1.0,2.0,3.0,4.0])
    x = [pi.normal(ai,bi) for ai, bi in zip([a[0], a[1], a[2], a[3], a[4]], [a[3], a[4], a[2], a[1], a[1]])]
    expected_a = {x[0]:0, x[1]:1, x[2]:2, x[3]:3, x[4]:4}
    expected_b = {x[0]:3, x[1]:4, x[2]:2, x[3]:1, x[4]:1}
    M = engine.run_all_vmaps(x)
    print_upstream([M[rv] for rv in x])

    vmap = M[x[0]].parents[0]
    assert vmap.op == VMap(Normal(), in_axes=[0, 0])
    assert rv_equal(vmap.parents[0], a)  

    for rv in x:
        assert M[rv].op == Index()
        assert M[rv].parents[0] == vmap  
        pos = M[rv].parents[1].op.value
        a_array = vmap.parents[0].op.value
        b_array = vmap.parents[1].parents[1].op.value
        assert a_array[pos] == expected_a[rv]
        assert b_array[pos] == expected_b[rv]

# test_same_parent()

def test_add_1d_vector():
    a = pi.constant([0.0,1.0,2.0])
    b = pi.constant([2.0,3.0,4.0])
    c = pi.constant([3.0,4.0,5.0])
    e = [a+b, b+c, a+c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])

    expected_valuea = {e[0]: a.op.value, e[1]: b.op.value, e[2]: a.op.value}
    expected_valueb = {e[0]: b.op.value, e[1]: c.op.value, e[2]: c.op.value}

    vmap = M[e[0]].parents[0]
    assert vmap.op == VMap(VMap(Add(), in_axes=[0,0], axis_size=3), in_axes=[0,0])

    for i, rv in enumerate(e):
        assert M[rv].parents[0] == vmap  
        a_array = vmap.parents[0].op.value
        b_array = vmap.parents[1].op.value
        assert np.array_equal(a_array[i], expected_valuea[rv])
        assert np.array_equal(b_array[i], expected_valueb[rv])

# test_add_1d_vector()

def test_matmul_1d_vector():
    a = pi.constant([0,1,2])
    b = pi.constant([2,3,4])
    c = pi.constant([3,4,5])
    e = [a@b, b@c, a@c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])

    expected_a = {e[0]: a.op.value, e[1]: b.op.value, e[2]: a.op.value}
    expected_b = {e[0]: b.op.value, e[1]: c.op.value, e[2]: c.op.value}

    vmap = M[e[0]].parents[0]
    assert vmap.op == VMap(Matmul(), in_axes=[0,0])

    for i, rv in enumerate(e):
        assert M[rv].op == Index()
        assert M[rv].parents[0] == vmap 
        assert M[rv].parents[1].op == Constant(i)
        a_array = vmap.parents[0].op.value
        b_array = vmap.parents[1].op.value
        assert np.array_equal(a_array[i], expected_a[rv])
        assert np.array_equal(b_array[i], expected_b[rv])

# test_matmul_1d_vector()

def test_add_2d_vector():
    a = pi.constant([[0,1,2], [3,4,5]])
    b = pi.constant([[2,3,4], [5,6,7]])
    c = pi.constant([[3,4,5], [6,7,8]])
    e = [a+b, b+c, a+c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])

    expected_a = {e[0]: a.op.value, e[1]: b.op.value, e[2]: a.op.value}
    expected_b = {e[0]: b.op.value, e[1]: c.op.value, e[2]: c.op.value}

    vmap = M[e[0]].parents[0]
    assert vmap.op == VMap(VMap(VMap(Add(), in_axes=[0,0], axis_size=3), in_axes=[0,0], axis_size=2), in_axes=[0,0])

    for i, rv in enumerate(e):
        assert M[rv].parents[0] == vmap 
        a_array = vmap.parents[0].op.value
        b_array = vmap.parents[1].op.value
        assert np.array_equal(a_array[i], expected_a[rv])
        assert np.array_equal(b_array[i], expected_b[rv])

# test_add_2d_vector()

def test_matmul_2d_vector():
    a = pi.constant([[0,1,2], [3,4,5], [6,7,8]])
    b = pi.constant([[2,3,4], [5,6,7], [8,9,10]])
    c = pi.constant([[3,4,5], [6,7,8], [9,10,11]])
    e = [a@b, b@c, a@c]
    M = engine.run_all_vmaps(e)
    print_upstream([M[rv] for rv in e])

    expected_a = {e[0]: a.op.value, e[1]: b.op.value, e[2]: a.op.value}
    expected_b = {e[0]: b.op.value, e[1]: c.op.value, e[2]: c.op.value}

    vmap = M[e[0]].parents[0]
    assert vmap.op == VMap(Matmul(), in_axes=[0,0])

    for i, rv in enumerate(e):
        assert M[rv].parents[0] == vmap  
        a_array = vmap.parents[0].op.value
        b_array = vmap.parents[1].op.value
        assert np.array_equal(a_array[i], expected_a[rv])
        assert np.array_equal(b_array[i], expected_b[rv])

# test_matmul_2d_vector()

def test_inner_product():
    a = pi.constant(0)
    b = pi.constant(1)
    c = pi.constant(3)
    d = pi.constant(4)
    e = pi.constant(5)
    f = pi.constant(6)
    g = a*b + c*d + e*f
    M = engine.run_all_vmaps([g])
    print_upstream(M[g])

    rv = M[g]
    assert rv.op == Add()

    vmap1 = rv.parents[1].parents[0]
    assert vmap1.op == VMap(Mul(), in_axes=[0,0])
    assert rv_equal(vmap1.parents[0], pi.constant([0,3,5]))
    assert rv_equal(vmap1.parents[1], pi.constant([1,4,6]))
    assert rv.parents[1].op == Index()
    assert rv.parents[1].parents[1].op == Constant(2)  

    assert rv.parents[0].op == Add()
    assert rv.parents[0].parents[1].op == Index()
    vmap2 = rv.parents[0].parents[1].parents[0]
    assert rv_equal(vmap1, vmap2)
    assert rv.parents[0].parents[1].parents[1].op == Constant(1)  

    assert rv.parents[0].parents[0].op == Index()
    vmap3 = rv.parents[0].parents[0].parents[0]
    assert rv_equal(vmap1, vmap3)
    assert rv.parents[0].parents[0].parents[1].op == Constant(0)  

# test_inner_product()

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
    assert rv1.parents[1].op == Constant(0)

    vmap_add1 = rv1.parents[0]
    assert vmap_add1.op == VMap(Add(), in_axes=[0,0])

    assert vmap_add1.parents[0].op == Index()
    assert rv_equal(vmap_add1.parents[0].parents[1], pi.constant([0, 2]))
    assert vmap_add1.parents[1].op == Index()
    assert rv_equal(vmap_add1.parents[1].parents[1], pi.constant([1, 3]))

    vmap_mul1 = vmap_add1.parents[1].parents[0]
    assert rv_equal(vmap_add1.parents[0].parents[0], vmap_mul1)  
    assert vmap_mul1.op == VMap(Mul(), in_axes=[0,0])
    assert rv_equal(vmap_mul1.parents[0], pi.constant([1,2,5,6]))
    assert rv_equal(vmap_mul1.parents[1], pi.constant([1,2,1,2]))

    rv2 = M[e2]
    assert rv2.op == Index()
    assert rv2.parents[1].op == Constant(1)
    vmap_add2 = rv2.parents[0]
    assert rv_equal(vmap_add2, vmap_add1)

# test_matrix_vector_product()

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
    assert rv1.op == Add()
    
    g = rv1.parents[0]
    i = rv1.parents[1]
    assert g.op == Index()
    assert i.op == Index()
    assert g.parents[1].op == Constant(0)
    assert i.parents[1].op == Constant(1)
    
    f = g.parents[0]
    assert f is i.parents[0]  
    assert f.op == Index()
    assert f.parents[1].op == Constant(0)  
    assert rv_equal(f.parents[2], pi.constant([0, 1])) 

    vmap_mul_0 = f.parents[0]
    assert vmap_mul_0.op == VMap(VMap(Mul(), in_axes=[0, 0]), in_axes=[0, None])
    assert rv_equal(vmap_mul_0.parents[0], pi.constant([[1,2],[5,6]]))
    assert rv_equal(vmap_mul_0.parents[1], pi.constant([1,2]))

    rv2 = M[e2]
    assert rv2.op == Add()
    l = rv2.parents[0]
    m = rv2.parents[1]
    assert l.op == Index()
    assert m.op == Index()
    assert l.parents[1].op == Constant(0)
    assert m.parents[1].op == Constant(1)
    k = l.parents[0]
    assert k is m.parents[0]
    assert k.op == Index()
    assert k.parents[1].op == Constant(1) 
    vmap_mul_1 = k.parents[0]
    assert rv_equal(vmap_mul_0, vmap_mul_1)

# test_matrix_vector_product_2()

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

    vmap_add = M[C_flat[0]].parents[0]
    assert vmap_add.op == VMap(Add(), in_axes=[0,0])

    left_gather = vmap_add.parents[0]
    right_gather = vmap_add.parents[1]
    assert left_gather.op == Index()
    assert right_gather.op == Index()
    vmap_mul = left_gather.parents[0]
    assert vmap_mul is right_gather.parents[0]
    assert vmap_mul.op == VMap(Mul(), in_axes=[0,0])

    a_vals = vmap_mul.parents[0].op.value
    b_vals = vmap_mul.parents[1].op.value
    left_idx = left_gather.parents[1].op.value
    right_idx = right_gather.parents[1].op.value

    dot_products = [(A[0][0], B[0][0], A[0][1], B[1][0]),
                    (A[0][0], B[0][1], A[0][1], B[1][1]),
                    (A[1][0], B[0][0], A[1][1], B[1][0]),
                    (A[1][0], B[0][1], A[1][1], B[1][1])]

    for i, (a1, b1_, a2, b2_) in enumerate(dot_products):
        pos1 = left_idx[i]
        assert a_vals[pos1] == a1.op.value
        assert b_vals[pos1] == b1_.op.value
        pos2 = right_idx[i]
        assert a_vals[pos2] == a2.op.value
        assert b_vals[pos2] == b2_.op.value

    for i, c in enumerate(C_flat):
        rv = M[c]
        assert rv.op == Index()
        assert rv.parents[0] == vmap_add
        assert rv.parents[1].op.value == i

    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[1]].parents[0])
    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[2]].parents[0])
    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[3]].parents[0])

# test_matrix_matrix_product()

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

    vmap_add = M[C_flat[0]].parents[0]
    assert vmap_add.op == VMap(Add(), in_axes=[0,0])

    idx_1 = vmap_add.parents[0]
    idx_2 = vmap_add.parents[1]
    assert idx_1.op == Index()
    assert idx_2.op == Index()

    vmap_mul_nested = idx_1.parents[0]
    assert vmap_mul_nested is idx_2.parents[0]
    assert vmap_mul_nested.op == VMap(VMap(Mul(), in_axes=[0,0]), in_axes=[1,0])

    for i, c in enumerate([c00, c01, c10, c11]):
        rv = M[c]
        assert rv.op == Index()
        assert rv.parents[0] == vmap_add
        assert rv.parents[1].op.value == i

    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[1]].parents[0])

# test_matrix_matrix_product_2()

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

    expected_mul = VMap(
        VMap(
            VMap(Mul(), in_axes=[0, None]), 
            in_axes=[1, 0]
        ), 
        in_axes=[None, 0]
    )

    for c in C_flat:
        rv = M[c]
        assert rv.op == Index()
        
        vmap_add = rv.parents[0]
        assert vmap_add.op == VMap(Add(), in_axes=[0,0])
        
        idx_1 = vmap_add.parents[0]
        idx_2 = vmap_add.parents[1]
        assert idx_1.op == Index()
        assert idx_2.op == Index()
        
        inter_idx_1 = idx_1.parents[0]
        inter_idx_2 = idx_2.parents[0]
        
        assert inter_idx_1.op == Index()
        assert rv_equal(inter_idx_1, inter_idx_2)
        
        vmap_mul_nested = inter_idx_1.parents[0]
        assert vmap_mul_nested.op == expected_mul
        assert rv_equal(vmap_mul_nested.parents[0], A)
        assert rv_equal(vmap_mul_nested.parents[1], pi.constant([[5,7],[6,8]]))

    assert rv_equal(M[C_flat[0]].parents[0], M[C_flat[1]].parents[0])
    assert rv_equal(M[C_flat[2]].parents[0], M[C_flat[3]].parents[0])

    inter_group_a = M[C_flat[0]].parents[0].parents[0].parents[0]
    inter_group_b = M[C_flat[2]].parents[0].parents[0].parents[0]
    assert not rv_equal(inter_group_a, inter_group_b)

# test_matrix_matrix_product_3()

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

    # Priors: one weight per (input_dim=1 -> hidden), one bias per hidden unit,
    # then hidden -> output weights/bias. All shared across the whole dataset.
    w1 = [pi.normal(0, 1) for _ in range(hidden)]        # hidden weights
    b1 = [pi.normal(0, 1) for _ in range(hidden)]        # hidden biases
    w2 = [pi.normal(0, 1) for _ in range(hidden)]        # output weights
    b2 = pi.normal(0, 1)                                  # output bias
    sigma = pi.constant(0.5)

    xs = [pi.constant(float(i)) for i in range(n_inputs)]

    ys = []
    for x in xs:
        # hidden layer: h_j = tanh(w1_j * x + b1_j) for j in range(hidden)
        hidden_acts = [pi.tanh(w1[j] * x + b1[j]) for j in range(hidden)]
        # output: sum_j w2_j * h_j + b2
        out = b2
        for j in range(hidden):
            out = out + w2[j] * hidden_acts[j]
        y = pi.normal(out, sigma)
        ys.append(y)

    M = engine.run_all_vmaps(ys)
    print_upstream([M[y] for y in ys])

test_bayesian_neural_network_stress()