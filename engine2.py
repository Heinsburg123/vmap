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

    def group_index(self, drv):
        if drv.op.name != "Index":
            return "NotIndex"
        own = []
        for i in range(1, len(drv.parents)):
            con = self.get_constant(drv.parents[i])
            if isinstance(con, str):
                return "NotIndex"
            own.append(con)
        if drv.parents[0].op.name != "Index":
            return own
        inner_lst = self.group_index(drv.parents[0])
        if inner_lst == "NotIndex":
            return "NotIndex"
        merged, j = [], 0
        for entry in inner_lst:
            arr = np.asarray(entry)
            if arr.ndim == 0:
                merged.append(entry)
            else:
                merged.append(arr[own[j]]); j += 1
        return merged

    def get_const_drv(self, arr):
        a = np.asarray(arr)
        key = (a.shape, a.dtype.str, a.tobytes())
        if key not in self.const_cache:
            self.const_cache[key] = DRV.from_rv(RV(Constant(a)))
        return self.const_cache[key]

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

    def _serialize(self, lst):
        parts = []
        for el in lst:
            arr = np.asarray(el)
            header = np.array([arr.ndim] + list(arr.shape), dtype=np.int64).tobytes()
            parts.append(header + arr.tobytes())
        return b"|".join(parts)

    def _deserialize(self, blob):
        result = []
        for part in blob.split(b"|"):
            if not part:
                continue
            ndim  = int(np.frombuffer(part[:8], dtype=np.int64)[0])
            shape = tuple(np.frombuffer(part[8:8+8*ndim], dtype=np.int64).tolist())
            data  = np.frombuffer(part[8+8*ndim:], dtype=np.int64).reshape(shape)
            result.append(data)
        return result

    def batch_constants(self, DRVs):
        shape_groups = {}
        for drv in DRVs:
            if drv.op.name == "Constant":
                shape_groups.setdefault(drv.shape, []).append(drv)
        for shape, group in shape_groups.items():
            if len(group) <= 1:
                continue
            group.sort(key=lambda d: d._n)
            stacked = self.get_const_drv(np.stack([d.op.value for d in group]))
            for i, drv in enumerate(group):
                drv.update(Index(), [stacked, self.get_const_drv(i)]
                           + [self.get_const_drv(list(range(d))) for d in shape])

    def build_buckets(self, DRVs):
        buckets, index_rv = {}, {}
        for drv in DRVs:
            if drv.op.name in ("Constant", "Index"):
                continue
            index_lst    = [self.group_index(p) for p in drv.parents]
            base_parents = tuple(self.get_resolved_parent(p) for p in drv.parents)
            index_rv[drv._n] = index_lst

            possible_axes = []
            for idd in index_lst:
                if isinstance(idd, str):
                    possible_axes.append(["None"])
                else:
                    cands = [j for j, v in enumerate(idd)
                             if not isinstance(v, str) and np.asarray(v).ndim == 0]
                    possible_axes.append(cands + ["None"])

            for combo in product(*possible_axes):
                remain = []
                for i, ax in enumerate(combo):
                    if ax == "None":
                        remain.append(index_lst[i] if isinstance(index_lst[i], str)
                                      else self._serialize(index_lst[i]))
                    else:
                        remain.append(self._serialize(index_lst[i][:ax] + index_lst[i][ax+1:]))
                key = (drv.op, *base_parents, drv._shape, *combo, *remain)
                buckets.setdefault(key, []).append(drv)

        return buckets, index_rv


    def build_new_parents(self, group, axes, remain, index_rv):
        new_parents, final_axes = [], []
        for i, ax in enumerate(axes):
            ax = None if ax == "None" else ax
            final_axes.append(ax)

            if isinstance(remain[i], str) and remain[i] == "NotIndex":
                new_parents.append(self.get_resolved_parent(group[0].parents[i]))
                continue

            ndim, idd, c_drvs = len(index_rv[group[0]._n][i]), 0, []
            for j in range(ndim):
                if j == ax:
                    arr = [index_rv[drv._n][i][j] for drv in group]
                else:
                    arr = remain[i][idd]; idd += 1
                c_drvs.append(self.get_const_drv(arr))

            parent = self.get_resolved_parent(group[0].parents[i])
            if self.indices_fill_parent(c_drvs, parent):
                new_parents.append(parent)
            else:
                new_parents.append(DRV.fresh(Index(), [parent] + c_drvs))

            if new_parents[-1].ndim == 1:
                final_axes[i] = 0 if final_axes[i] is not None else None

        return new_parents, final_axes


    def run_vmap(self, DRVs):

        buckets, index_rv = self.build_buckets(DRVs)
        if not buckets:
            return False

        def priority(item):
            key, group = item
            k = len(index_rv[group[0]._n])
            none_count = sum(1 for ax in key[2+k : 2+2*k] if ax == "None")
            return (len(group), none_count)

        best_key, group = max(buckets.items(), key=priority)
        if len(group) <= 1:
            return False

        k      = len(index_rv[group[0]._n])
        axes   = list(best_key[2+k : 2+2*k])
        remain = [p if isinstance(p, str) else self._deserialize(p)
                  for p in best_key[2+2*k:]]

        group.sort(key=lambda drv: tuple(
            int(index_rv[drv._n][i][axes[i]])
            for i in range(len(axes)) if axes[i] != "None"
        ))

        new_parents, final_axes = self.build_new_parents(group, axes, remain, index_rv)

        tensor_axes = tuple(
            None if ax is None else
            sum(1 for e in remain[i][:ax] if np.asarray(e).ndim == 1)
            for i, ax in enumerate(final_axes)
        )

        need_axis_size = all(a is None for a in final_axes)
        op       = VMap(best_key[0], in_axes=tensor_axes,
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
        self.batch_constants(DRVs)

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