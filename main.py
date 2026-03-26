from pangolin.ir import * 
from engine import VmapEngine

a = RV(Constant([[1,2,3], [2,3,3]]))
b = RV(Constant(0))
c = RV(Constant(1))
d = RV(Index(), a, b, c)
e = RV(Index(), a, b, b)
f = RV(Index(), a, c, b)
g1 = RV(Add(), d, e)
g2 = RV(Add(), e, f)

VmapEngine().run_vmap([a, b, c, d, e, f, g1, g2])
