from itertools import product
import numpy as np
from pangolin.ir import *
from pangolin.dag import *
from DRV import DRV

class VmapEngine:    
    def get_resolved_parent(self, drv):
        while drv.op.name == "Index":
            drv = drv.parents[0]
        return drv

    def get_constant(self, drv):
        if drv.op.name == "Constant":
            return drv.op.value
        if drv.op.random:
            return "Random"
        lst = []
        for i in range(len(drv.parents) - 1, 0, -1):
            con = self.get_constant(drv.parents[i])
            if isinstance(con, str):
                return "Random"
            lst.append(con)
        p = self.get_resolved_parent(drv)
        if p.op.random:
            return "Random"
        lst.reverse()
        return p.op.value[tuple(lst)]

    def get_const_drv(self, arr):
        a = np.asarray(arr)
        key = (a.shape, a.dtype.str, a.tobytes())
        if key not in self.const_cache:
            self.const_cache[key] = DRV.from_rv(RV(Constant(a)))
        return self.const_cache[key]

    def _to_hashable(self, v):
        a = np.asarray(v)
        return int(a) if a.ndim == 0 else tuple(a.tolist())

    def indices_fill_parent(self, c_drvs, parent):
        if len(c_drvs) != len(parent.shape):
            return False
        for c_drv, dim_size in zip(c_drvs, parent.shape):
            val = np.asarray(c_drv.op.value)
            if val.ndim != 1 or len(val) != dim_size:
                return False
            if not np.array_equal(val, np.arange(dim_size)):
                return False
        return True
        
    def compute_hash_keys(self, drv):
        if drv.op.name in ("Constant", "Index"):
            return []

        axes_options = []
        drv_indices  = []

        for p in drv.parents:
            if p.op.name == "Constant":
                axes_options.append(["None"])
                drv_indices.append("NotIndex")

            elif p.op.name == "Index":
                idx = [self.get_constant(p.parents[k])
                       for k in range(1, len(p.parents))]
                scalar_pos = [k for k in range(len(idx))
                                if np.asarray(idx[k]).ndim == 0]
                axes_options.append(scalar_pos + ["None"])
                drv_indices.append(idx)

            else:
                axes_options.append(["None"])
                drv_indices.append("NotIndex")

        self.index_rv[drv._n] = drv_indices

        hash_keys = []
        for lst in product(*axes_options):
            parent_key = []
            for i, p in enumerate(drv.parents):
                idd = drv_indices[i]
                ax  = lst[i]
                if idd == "NotIndex":
                    if p.op.name == "Constant":
                        parent_key.append(("Const", tuple(p.op.value.shape)))
                    else:
                        parent_key.append(p)
                else:
                    if ax == "None":
                        remain_key = tuple(self._to_hashable(idd[k])
                                           for k in range(len(idd)))
                    else:
                        remain_key = tuple(self._to_hashable(idd[k])
                                           for k in range(len(idd)) if k != ax)
                    parent_key.append((p.parents[0], ax, remain_key))

            hash_keys.append((drv.op, *parent_key))

        return hash_keys

    def build_buckets(self, DRVs):
        self.index_rv = {}
        buckets = {}

        for drv in DRVs:
            for hash_key in self.compute_hash_keys(drv):
                buckets.setdefault(hash_key, []).append(drv)

        return buckets
        
    def build_new_parents(self, key, group):
        new_parents = []
        final_axes  = []

        for i, pk in enumerate(key[1:]):
            if not isinstance(pk, tuple):
                new_parents.append(group[0].parents[i])
                final_axes.append(None)
            elif pk[0] == "Const":
                stacked = np.stack([np.asarray(group[j].parents[i].op.value)
                                    for j in range(len(group))])
                if(all(np.array_equal(x, stacked[0]) for x in stacked)):
                    stacked = stacked[0]
                    if stacked.ndim == 1 and len(stacked) == 1:
                        stacked = stacked.item()
                    final_axes.append(None)
                else:
                    final_axes.append(0)
                new_parents.append(self.get_const_drv(stacked))
            else:
                base, ax, remain_tuple = pk
                if ax == "None":
                    new_parents.append(group[0].parents[i])
                    final_axes.append(None)
                else:
                    ndim = len(self.index_rv[group[0]._n][i])
                    remain_cur, c_drvs = 0, []
                    for j in range(ndim):
                        if j == ax:
                            arr = [self.index_rv[drv._n][i][j] for drv in group]
                        else:
                            arr = np.asarray(remain_tuple[remain_cur])
                            remain_cur += 1
                        c_drvs.append(self.get_const_drv(arr))
 
                    if self.indices_fill_parent(c_drvs, base):
                        new_parents.append(base)
                    else:
                        new_parents.append(DRV.fresh(Index(), [base] + c_drvs))

                    tensor_ax = sum(1 for j in range(ax)
                                    if isinstance(remain_tuple[j], tuple))
                    if new_parents[-1].ndim == 1:
                        tensor_ax = 0
                    final_axes.append(tensor_ax)
        return new_parents, final_axes

    def run_vmap(self, DRVs):
        buckets = self.build_buckets(DRVs)
        if not buckets:
            return False
 
        def priority(item):
            key, group = item
            none_count = sum(
                1 for pk in key[1:]
                if not isinstance(pk, tuple)                      
                or (len(pk) == 3 and pk[1] == "None")          
            )
            return (len(group), none_count)
 
        best_key, group = max(buckets.items(), key=priority)
        if len(group) <= 1:
            return False
 
        def sort_key(drv):
            return tuple(
                int(self.index_rv[drv._n][i][pk[1]])
                for i, pk in enumerate(best_key[1:])
                if isinstance(pk, tuple) and len(pk) == 3 and pk[1] != "None"
            )
        group.sort(key=sort_key)
 
        new_parents, final_axes = self.build_new_parents(best_key, group)
 
        need_axis_size = all(a is None for a in final_axes)
        op = VMap(best_key[0], in_axes=tuple(final_axes),
                  **({"axis_size": len(group)} if need_axis_size else {}))
        vmap_drv = DRV.fresh(op, new_parents)
        DRVs.append(vmap_drv)
 
        for idx, drv in enumerate(group):
            drv.update(Index(), [vmap_drv, self.get_const_drv(idx)]
                       + [self.get_const_drv(list(range(d))) for d in vmap_drv.shape[1:]])
        return True

    def run_all_vmaps(self, RVs):
        self.const_cache = {}
        DRV.memo.clear()

        DRVs = [DRV.from_rv(rv) for rv in upstream_nodes(RVs)]

        while self.run_vmap(DRVs):
            pass

        rv_cache = {}
        def to_rv(drv):
            if drv._n in rv_cache:
                return rv_cache[drv._n]
            rv = RV(drv.op, *[to_rv(p) for p in drv.parents])
            rv_cache[drv._n] = rv
            return rv

        for drv in DRVs:
            to_rv(drv)
        return {rv: rv_cache[rv._n] for rv in RVs}