import numpy as np
from pangolin.ir import *
from pangolin.dag import *

class DRV:
    memo = {}
    _counter = -1  # negative to avoid collision with RV._n

    @classmethod
    def _next_n(cls):
        cls._counter -= 1
        return cls._counter

    @classmethod
    def from_rv(cls, rv):
        if rv in cls.memo:
            return cls.memo[rv]
        drv = cls.__new__(cls)
        cls.memo[rv] = drv
        drv._shape = rv.shape
        drv._n = rv._n
        drv.op = rv.op
        drv.parents = [cls.from_rv(p) for p in rv.parents]
        return drv

    @classmethod
    def fresh(cls, op, parents):
        """Create a new DRV not backed by any original RV."""
        drv = cls.__new__(cls)
        drv.op = op
        drv.parents = list(parents)
        drv._shape = op.get_shape(*tuple(p.shape for p in parents))
        drv._n = cls._next_n()
        return drv

    def update(self, op, parents):
        self.op = op
        self.parents = list(parents)
        self._shape = op.get_shape(*tuple(p.shape for p in parents))

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __repr__(self):
        ret = "DRV(" + repr(self.op)
        for p in self.parents:
            ret += ", " + repr(p)
        return ret + ")"