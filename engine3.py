from itertools import product
import numpy as np
from pangolin.ir import *
from pangolin.dag import *
from DRV import DRV
from bucket_heap import BucketHeap, _priority

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

        # Given/observed-status separation: two RVs that would otherwise
        # hash identically (same op, same parent structure) must NOT be
        # grouped into the same VMap bucket if one is observed data and the
        # other is a free latent -- fusing them would make it impossible to
        # condition on just the observed subset afterward (the fused node
        # would have no single consistent "given value" to assign). Adding
        # this as a factor of the hash key ensures observed/unobserved RVs
        # can never collide into the same bucket, while leaving all other
        # grouping logic (parent/axis structure) untouched.
        is_given = drv._n in getattr(self, "given_ns", ())

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

            hash_keys.append((drv.op, is_given, *parent_key))

        return hash_keys

    def build_new_parents(self, key, group):
        new_parents = []
        final_axes  = []

        for i, pk in enumerate(key[2:]):
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


    def run_vmap(self, DRVs, heap, best_key, group):
        def sort_key(drv):
            return tuple(
                int(self.index_rv[drv._n][i][pk[1]])
                for i, pk in enumerate(best_key[2:])
                if isinstance(pk, tuple) and len(pk) == 3 and pk[1] != "None"
            )
        group = sorted(group, key=sort_key)

        new_parents, final_axes = self.build_new_parents(best_key, group)

        need_axis_size = all(a is None for a in final_axes)
        op = VMap(best_key[0], in_axes=tuple(final_axes),
                  **({"axis_size": len(group)} if need_axis_size else {}))
        vmap_drv = DRV.fresh(op, new_parents)
        DRVs.append(vmap_drv)

        touched = heap.dependents_of(group)

        heap.remove_group(group)

        for idx, drv in enumerate(group):
            drv.update(Index(), [vmap_drv, self.get_const_drv(idx)]
                       + [self.get_const_drv(list(range(d))) for d in vmap_drv.shape[1:]])

        heap.add_drv(vmap_drv)

        heap.refresh(touched)

        return vmap_drv

    def run_all_vmaps(self, RVs, given_map=None):
        given_map = given_map or {}
        self.const_cache = {}
        self.index_rv = {}
        DRV.memo.clear()

        DRVs = [DRV.from_rv(rv) for rv in upstream_nodes(RVs)]

        # Resolve each given/observed RV to its corresponding DRV node id.
        # This must happen AFTER the DRVs above are built (so DRV.memo is
        # populated), so that DRV.from_rv(rv) here returns the SAME
        # (memoized) DRV instance already present in `DRVs`, rather than a
        # fresh orphan. compute_hash_keys uses this set to keep observed
        # and unobserved RVs from ever being grouped into the same bucket.
        self.given_ns = {DRV.from_rv(rv)._n for rv in given_map}

        heap = BucketHeap(self)
        heap.build(DRVs)

        while True:
            best_key, group = heap.pop_max()
            if best_key is None:
                break
            self.run_vmap(DRVs, heap, best_key, list(group))

        rv_cache = {}
        def to_rv(drv):
            if drv._n in rv_cache:
                return rv_cache[drv._n]
            rv = RV(drv.op, *[to_rv(p) for p in drv.parents])
            rv_cache[drv._n] = rv
            return rv

        for drv in DRVs:
            to_rv(drv)
            
        return {rv: rv_cache[rv._n] for rv in upstream_nodes(RVs)}