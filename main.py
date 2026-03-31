from pangolin.ir import *
from engine import VmapEngine

a = RV(Constant([[1,2,3],[4,5,6]]))
i0 = RV(Constant(0))
i1 = RV(Constant(1))

x = RV(Index(), a, i0, i0)
y = RV(Index(), a, i0, i1)
z = RV(Index(), a, i1, i0)

g1 = RV(Add(), x, y)
g2 = RV(Add(), x, z)
m1 = RV(Mul(), x, y)
m2 = RV(Mul(), x, z)

VmapEngine().run_vmap([a, i0, i1, x, y, z, g1, g2, m1, m2])