from pangolin.ir import * 
from engine import VmapEngine

a = RV(Constant([1,2,3]))
b = RV(Constant(1))
c = RV(SimpleIndex(), a, b)
d = RV(Add(), c, b)

print(VmapEngine().run_vmap([a, b, c, d]))

