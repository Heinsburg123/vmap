from pangolin.ir import * 
from engine import VmapEngine

a = RV(Constant([1,2,3]))
b = RV(Constant(1))
c = RV(Index(), a, b)
d = RV(Add(), c, b)
g = RV(Constant(2))
f = RV(Index(), a, g)
e = RV(Add(), f, b)

VmapEngine().run_vmap([a, b, c, d, e, f, g])
