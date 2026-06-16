from itertools import product
import numpy as np
import heapq
from pangolin.ir import * 
from collections import deque
import jax.numpy as jnp
from pangolin.dag import *

class VmapEngine:
    adj = {}
    visited = {}
    def get_resolved_parent(self, rv):
        tmp = rv
        while(tmp.op.name == "Index"):
            tmp = tmp.parents[0]
        return tmp

    def DFS(self, rv):
        for p in rv.parents:
            tmp = self.get_resolved_parent(p)
            if tmp not in self.adj:
                self.adj[tmp] = []
            self.adj[tmp].append(rv)
            if tmp not in self.visited:
                self.visited[tmp] = True
                self.DFS(tmp)

    def level_ranking(self, RVs):
        self.adj = {}
        self.visited = {}
        rank = {}
        in_degree = {}
        queue = deque()
        bucket = {}
        order_bucket = []
        for rv in RVs:
            if(rv not in self.visited and rv.op.name != "Index"):
                self.visited[rv] = True
                self.adj[rv] = []
                self.DFS(rv)

        rv_set = {rv for rv in RVs if rv.op.name != "Index"}

        for rv in RVs:
            if(rv.op.name == "Constant"):
                in_degree[rv] = 0
                queue.append(rv)
                rank[rv] = 0
            elif(rv.op.name != "Index"):
                internal_parents = 0
                for p in rv.parents:
                    tmp = self.get_resolved_parent(p)
                    if tmp in rv_set:
                        internal_parents += 1
                in_degree[rv] = internal_parents
                if internal_parents == 0:
                    queue.append(rv)
                    rank[rv] = 0

        while(queue):
            u = queue.popleft()
            for v in self.adj[u]:
                in_degree[v] -= 1
                if(in_degree[v] == 0):
                    for p in v.parents:
                        tmp = self.get_resolved_parent(p)
                        if tmp not in rank:
                            continue
                        rank[v] = max(rank[v],rank[tmp]+1) if v in rank else rank[tmp]+1
                    if v not in rank:
                        rank[v] = 0
                    queue.append(v)

        for rv in rank:
            if(rank[rv] not in bucket):
                bucket[rank[rv]] = []
            bucket[rank[rv]].append(rv)
        return [rvs for _, rvs in sorted(bucket.items())]
    
    def get_constant(self, rv):
        if(rv.op.name == "Constant"):
            return rv.op.value
        if(rv.op.random == True):
            return "Random"
        lst = []
        for i in range(len(rv.parents)-1, 0, -1):
            p = rv.parents[i]
            con = self.get_constant(p)
            lst.append(con)
            if(isinstance(con, str) and con == "Random"):
                return "Random"
        p = self.get_resolved_parent(rv)
        if(p.op.random == True):
            return "Random"
        lst.reverse()
        return p.op.value[tuple(lst)]

    def group_index(self, rv):
        if rv.op.name != "Index":
            return "NotIndex"
        own = []
        for i in range(1, len(rv.parents)):
            con = self.get_constant(rv.parents[i])
            if isinstance(con, str) and con == "Random":
                return "NotIndex"
            own.append(con)
        if rv.parents[0].op.name != "Index":
            return own
        inner_lst = self.group_index(rv.parents[0])
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
    
    def compute_hash(self, rv):
        base_parents = [self.get_resolved_parent(p) for p in rv.parents]
        return tuple([rv.op] + base_parents + [rv._shape])

    def deep_hash(self, rv, index_lst):
        ans = []
        possible_axes = []
        for idd in index_lst:
            if isinstance(idd, str):
                possible_axes.append(["None"])
                continue
            candidates = [j for j, val in enumerate(idd) if not isinstance(val, str) and val.ndim == 0]
            candidates.append("None")
            possible_axes.append(candidates)
        
        for combination in product(*possible_axes):
            ans.append(list(combination))

        def _serialize_index_list(lst):
            parts = []
            for el in lst:
                arr = np.asarray(el)
                header = np.array([arr.ndim] + list(arr.shape), dtype=np.int64).tobytes()
                parts.append(header + arr.tobytes())
            return b"|".join(parts)

        remaining = []
        for comb in ans:
            less = []
            for i,idx in enumerate(comb):
                if(idx == "None"):
                    if(isinstance(index_lst[i], str)):
                        less.append(index_lst[i])
                    else:
                        less.append(_serialize_index_list(index_lst[i]))
                else:
                    less.append(_serialize_index_list(index_lst[i][:idx] + index_lst[i][idx+1:]))
            remaining.append(less)
        return ans, remaining

    def run_greedy_set(self, universe, sets):
        uncovered = set(universe)
        for key in sets:
            sets[key] = set(sets[key])

        def none_count(key):
            return -sum(1 for el in key[:len(key)//2] if el == "None")

        heap = []
        counter = 0
        for key in sets:
            heap.append((-len(sets[key]), none_count(key), counter, key))
            counter += 1
        heapq.heapify(heap)
        result = {}         
        while uncovered and heap:
            while heap:
                neg_count, _, _, key = heapq.heappop(heap)   
                real_count = len(sets[key] & uncovered)
                if real_count == 0:
                    continue
                if real_count == -neg_count:
                    break
                heapq.heappush(heap, (-real_count, none_count(key), counter, key))
                counter += 1
            else:
                break
            contribution = sets[key] & uncovered       
            uncovered   -= contribution
            result[key] = contribution
        return result

    def run_vmap(self, RVs):    
        hash_map = {}
        index_rv = {}
        M = {} 
        const_cache = {}
        def get_const_rv(arr):
            a = np.asarray(arr)
            cache_key = (a.shape, a.dtype.str, a.tobytes())
            if cache_key not in const_cache:
                const_cache[cache_key] = RV(Constant(arr))
            return const_cache[cache_key]

        def indices_fill_parent(c_rvs, parent):
            shape = parent.shape
            if len(c_rvs) != len(shape):
                return False
            for idx_rv, dim_size in zip(c_rvs, shape):
                val = np.asarray(idx_rv.op.value)
                if val.ndim != 1 or len(val) != dim_size:
                    return False
                if not np.array_equal(val, np.arange(dim_size)):
                    return False
            return True

        for rv in RVs:
            if rv.op.name == "Constant" or rv.op.name == "Index":
                M[rv] = rv
        for rv in RVs:
            if(rv.op.name == "Constant" or rv.op.name == "Index"):
                continue
            hash_key = self.compute_hash(rv)
            if hash_key not in hash_map:
                hash_map[hash_key] = []
            hash_map[hash_key].append(rv)

        for key in hash_map:
            bucket = {}
            for rv in hash_map[key]:
                index = []
                for p in rv.parents:
                    index.append(self.group_index(p))
                index_rv[rv] = index
                get_axes, remaining = self.deep_hash(rv, index)
                for i in range(len(get_axes)):
                    axes = get_axes[i]
                    rmd = remaining[i]
                    tmp = tuple(axes+rmd)
                    if(tmp not in bucket):
                        bucket[tmp] = []
                    bucket[tmp].append(rv)

            final_bucket = self.run_greedy_set(hash_map[key], bucket)
            for key2 in final_bucket:
                final_bucket[key2] = list(final_bucket[key2])
            for key2 in final_bucket:
                if(len(final_bucket[key2]) == 1):
                    M[final_bucket[key2][0]] = final_bucket[key2][0]
                    continue
                axes = []
                remain = []
                c_rv = []
                axis_size = 0
                for i in range(len(key2)//2):
                    axes.append(key2[i])

                def _sort_key(rv):
                    return tuple(
                        int(index_rv[rv][i][axes[i]])
                        for i in range(len(axes))
                        if axes[i] != "None"
                    )
                final_bucket[key2].sort(key=_sort_key)

                need_axis_size = all(el == "None" for  el in axes)
                if(need_axis_size):
                    axis_size = len(final_bucket[key2])
                def _deserialize_index_list(blob):
                    result = []
                    for part in blob.split(b"|"):
                        if not part:
                            continue
                        ndim  = int(np.frombuffer(part[:8], dtype=np.int64)[0])
                        shape = tuple(np.frombuffer(part[8:8+8*ndim], dtype=np.int64).tolist())
                        data  = np.frombuffer(part[8+8*ndim:], dtype=np.int64).reshape(shape)
                        result.append(data)
                    return result

                for i in range(len(key2)//2, len(key2)):
                    if(isinstance(key2[i], str)):
                        remain.append(key2[i])
                    else:
                        remain.append(_deserialize_index_list(key2[i]))
                for i in range(len(key2)//2):
                    ndim = len(index_rv[final_bucket[key2][0]][i])
                    idd = 0
                    if(isinstance(remain[i], str) and remain[i] == "NotIndex"):
                        c_rv.append("NotIndex")
                        continue
                    c_rv.append([])
                    for j in range(ndim):
                        arr = []
                        if(j == axes[i]):
                            for rv in final_bucket[key2]:
                                arr.append(index_rv[rv][i][j])
                        else:
                            arr = remain[i][idd]
                            idd+=1
                        new_rv = get_const_rv(arr)  
                        c_rv[-1].append(new_rv)

                new_p = []
                for i in range(len(key2)//2):
                    args = []
                    if(isinstance(c_rv[i], str)):
                        new_p.append(key[i+1])
                    else:
                        parent = key[i+1]
                        if indices_fill_parent(c_rv[i], parent):
                            new_p.append(parent)
                        else:
                            args = [Index(), parent]
                            for var in c_rv[i]:
                                args.append(var)
                            new_p.append(RV(*args))

                    axes[i] = None if axes[i] == "None" else axes[i]

                    if(new_p[i].ndim == 1):
                        axes[i] = 0 if axes[i] is not None else None

                tensor_axes = []
                for i in range(len(key2)//2):
                    a = axes[i]
                    if a is None:
                        tensor_axes.append(None)
                    else:
                        tensor_axes.append(sum(1 for e in remain[i][:a] if np.asarray(e).ndim == 1))
                if(need_axis_size):
                    axis_size = len(final_bucket[key2])
                    op = VMap(key[0], in_axes=tuple(tensor_axes), axis_size=axis_size)
                else:
                    op = VMap(key[0], in_axes=tuple(tensor_axes))
                final_args = [op]
                for x in new_p:
                    final_args.append(x)
                vmap = RV(*final_args)
                print(f"Created vmap: {vmap}")
                for i, original_rv in enumerate(final_bucket[key2]):
                    index_args = [Index(), vmap, get_const_rv(i)]  # optimization 2
                    for dim_size in vmap.shape[1:]: 
                        index_args.append(get_const_rv(list(range(dim_size))))  # optimization 2
                    M[original_rv] = RV(*index_args)
        return M

    def batch_constants(self, RVs):
        M = {}
        shape_groups = {}
        for rv in RVs:
            if rv.op.name == "Constant":
                key = rv.shape  
                if key not in shape_groups:
                    shape_groups[key] = []
                shape_groups[key].append(rv)

        for shape, cc in shape_groups.items():
            if len(cc) > 1:
                rv_list = sorted(cc, key=lambda x: x._n)
                tmp = [rv.op.value for rv in rv_list]
                new_const = RV(Constant(tmp))
                for i, rv in enumerate(rv_list):
                    index_args = [Index(), new_const, RV(Constant(i))]
                    for dim_size in rv.shape:
                        index_args.append(RV(Constant(list(range(dim_size)))))
                    M[rv] = RV(*index_args)
                    
        return M

    def substitute_parents(self, rv, M):
        if rv in M:
            return M[rv]
        new_parents = []
        changed = False
        for p in rv.parents:
            new_p = self.substitute_parents(p, M)
            new_parents.append(new_p)
            if new_p is not p:
                changed = True
        if not changed:
            return rv
        return RV(rv.op, *new_parents)

    def run_all_vmaps(self, RVs):
        RVs = upstream_nodes(RVs)
        order_bucket = self.level_ranking(RVs)
        M = self.batch_constants(RVs)

        for group in order_bucket:
            sub_to_orig = {}
            substituted_group = []

            for rv in group:
                sub_rv = self.substitute_parents(rv, M)
                sub_to_orig[sub_rv] = rv
                substituted_group.append(sub_rv)

            group_M = self.run_vmap(substituted_group)

            while True:
                frontier = set()
                for replacement in group_M.values():
                    node = self.get_resolved_parent(replacement)
                    if node not in group_M:
                        frontier.add(node)

                if not frontier:
                    break

                M_next = self.run_vmap(list(frontier))
                made_progress = any(M_next.get(rv, rv) is not rv for rv in frontier)

                for orig in list(group_M.keys()):
                    group_M[orig] = self.substitute_parents(group_M[orig], M_next)
                group_M.update(M_next)

                if not made_progress:
                    break

            for sub_rv, orig_rv in sub_to_orig.items():
                M[orig_rv] = group_M.get(sub_rv, sub_rv)
            for node, replacement in group_M.items():
                if node not in sub_to_orig:
                    M[node] = replacement

        return M