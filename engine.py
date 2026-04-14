from itertools import product
import numpy as np
import heapq
from pangolin.ir import * 
from collections import deque

class VmapEngine:
    adj = {}
    visited = {}

    def DFS(self, rv):
        for p in rv.parents:
            tmp = p
            while(tmp.op.name == "Index"):
                tmp = tmp.parents[0]
            # Always add the edge parent -> rv so BFS can propagate ranks,
            # even when tmp was already registered by an earlier DFS call.
            if tmp not in self.adj:
                self.adj[tmp] = []
            self.adj[tmp].append(rv)
            # Only recurse into tmp if unvisited; prevents infinite loops
            # without suppressing edges to already-registered parents.
            if tmp not in self.visited:
                self.visited[tmp] = True
                self.DFS(tmp)

    def level_ranking(self, RVs):
        # Reset per-call so repeated calls don't accumulate stale edges.
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
        # Build the set of RVs that participate in the ranking so we can
        # count in-degree only from parents that are actually inside this set.
        # Parents outside the set (e.g. original data nodes when ranking a
        # frontier of new vmap nodes) must not inflate in-degree, otherwise
        # those nodes never reach zero and are silently dropped.
        rv_set = {rv for rv in RVs if rv.op.name != "Index"}

        for rv in RVs:
            if(rv.op.name == "Constant"):
                in_degree[rv] = 0
                queue.append(rv)
                rank[rv] = 0
            elif(rv.op.name != "Index"):
                # Count only parents whose resolved base node is in rv_set.
                internal_parents = 0
                for p in rv.parents:
                    tmp = p
                    while tmp.op.name == "Index":
                        tmp = tmp.parents[0]
                    if tmp in rv_set:
                        internal_parents += 1
                in_degree[rv] = internal_parents
                # Node with no in-set parents is a root: treat as rank 0.
                if internal_parents == 0:
                    queue.append(rv)
                    rank[rv] = 0

        while(queue):
            u = queue.popleft()
            for v in self.adj[u]:
                in_degree[v] -= 1
                if(in_degree[v] == 0):
                    for p in v.parents:
                        tmp = p
                        if(p.op.name == "Index"):
                            tmp = tmp.parents[0]
                        # Skip parents outside the current frontier (not in rank).
                        # External nodes are already computed; their effective rank
                        # is below everything in this pass, so they don't constrain
                        # the ordering here.
                        if tmp not in rank:
                            continue
                        rank[v] = max(rank[v],rank[tmp]+1) if v in rank else rank[tmp]+1
                    # If all parents were external, treat this node as a root.
                    if v not in rank:
                        rank[v] = 0
                    queue.append(v)

        for rv in rank:
            if(rank[rv] not in bucket):
                bucket[rank[rv]] = []
            bucket[rank[rv]].append(rv)

        for i in range(0, max(bucket.keys())+1):
            if(i in bucket):
                order_bucket.append(bucket[i])
        return order_bucket

    def group_index(self, rv):
        lst = []
        if(rv.op.name != "Index"):
            return "NotIndex"
        while(rv.op.name == "Index"):
            for i in range(len(rv.parents)-1, 0, -1):
                lst.append(rv.parents[i])
            rv = rv.parents[0]
        lst.reverse()
        ans = []
        for idd in lst:
            if(idd.op.name == "Constant"):
                ans.append(idd.op.value)
            else:
                ans.append("v_"+str(idd._n))
        return ans
    
    def compute_hash(self, rv):
        lst = [rv.op]
        for p in rv.parents:
            tmp = p
            while(tmp.op.name == "Index"):
                tmp = tmp.parents[0]
            lst.append(tmp)
        lst.append(rv._shape)
        return tuple(lst)

    def deep_hash(self, rv, index_lst):
        ans = []
        possible_axes = []
        for idd in index_lst:
            possible_axes.append([])
            if(isinstance(idd, str)):
                possible_axes[-1].append("None")
                continue
            #need to deal with instance where index might be RV
            for j in range(len(idd)):
                if idd[j].ndim == 0:
                    possible_axes[-1].append(j)
            possible_axes[-1].append("None")
        
        # Generate all combinations by selecting one element from each vector
        for combination in product(*possible_axes):
            ans.append(list(combination))
        def _serialize_index_list(lst):
            # Serialize each element individually to handle mixed 0-d/1-d shapes.
            # Layout per element: [ndim: int64][shape dims: int64*ndim][data bytes]
            parts = []
            for el in lst:
                arr = np.asarray(el)
                header = np.array([arr.ndim] + list(arr.shape), dtype=np.int64).tobytes()
                parts.append(header + arr.tobytes())
            return b"|".join(parts)

        remaining = []
        for comb in ans:
            less = []
            #This is trying to get the remaining index pattern after 
            # removing the axis that we are batching on
            for i,idx in enumerate(comb):
                if(idx == "None"):
                    if(isinstance(index_lst[i], str)):
                        less.append(index_lst[i])
                    else:
                        less.append(_serialize_index_list(index_lst[i]))
                else:
                    less.append(_serialize_index_list(
                        index_lst[i][:idx] + index_lst[i][idx+1:]
                    ))
            remaining.append(less)
        return ans, remaining

    def run_greedy_set(self, universe, sets):
        uncovered = set(universe)
        for key in sets:
            sets[key] = set(sets[key])
        heap = []
        counter = 0
        for key in sets:
            heap.append((-len(sets[key]), counter, key))
            counter += 1
        heapq.heapify(heap)
        result = {}
        while uncovered and heap:
            # ---- lazy-deletion loop ----
            while heap:
                neg_count, _, key = heapq.heappop(heap)   # O(log m)
                real_count = len(sets[key] & uncovered)
                if real_count == 0:
                    continue
                if real_count == -neg_count:
                    break
                heapq.heappush(heap, (-real_count, counter, key))
                counter += 1
            else:
                break
            contribution = sets[key] & uncovered        # new elements only
            uncovered   -= contribution
            result[key] = contribution
        return result


    def run_vmap(self, RVs):    
        hash_map = {}
        index_rv = {}
        M = {}  # Maps original RV -> replacement RV
        # Passthroughs: Constants and Index RVs map to themselves
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
        #This part is going through all the most general hash that is the Ops and parents
        for key in hash_map:
            bucket = {}
            for rv in hash_map[key]:
                index = []
                #Here is when we are going through the parents of the RV and getting 
                # the index pattern for each parent and storing it in index  
                for p in rv.parents:
                    index.append(self.group_index(p))
                #The index_rv dictionary helps us later to try and get the indexing pattern
                #for each RV 
                index_rv[rv] = index
                #Now we have the index pattern for each parent of the RV and the remaining
                # index pattern if we were to batch on a certain axis. Now 
                # we want to group them by the possible axes that we can batch on.
                get_axes, remaining = self.deep_hash(rv, index)
                #Going through all the possible axes or hash 
                for i in range(len(get_axes)):
                    axes = get_axes[i]
                    rmd = remaining[i]
                    #tmp is now the key for the bucket that we are grouping the RVs into.
                    #It consists of the axes that we are batching on and the remaining 
                    # index pattern after removing those axes
                    tmp = tuple(axes+rmd)
                    if(tmp not in bucket):
                        bucket[tmp] = []
                    bucket[tmp].append(rv)
            #Run greedy set cover on the buckets to find the best grouping 
            # of RVs to batch together
            final_bucket = self.run_greedy_set(hash_map[key], bucket)
            for key2 in final_bucket:
                final_bucket[key2] = list(final_bucket[key2])
            for key2 in final_bucket:
                if(len(final_bucket[key2]) == 1):
                    # Singleton: no batching, original RV maps to itself
                    M[final_bucket[key2][0]] = final_bucket[key2][0]
                    continue
                axes = []
                remain = []
                c_rv = []
                axis_size = 0
                #Get the axes that we are batching on 
                for i in range(len(key2)//2):
                    axes.append(key2[i])

                # Sort the group by the batched index values so that all vmaps
                # covering the same logical structure (e.g. every column-group
                # has rows [0,1,2]) end up with identical internal orderings.
                # Without this, the greedy set cover may yield e.g. [2,0,1]
                # for one column-group and [0,1,2] for another, making their
                # "remaining" byte-strings differ and preventing run_to_fixpoint
                # from batching them together in the next iteration.
                def _sort_key(rv):
                    return tuple(
                        int(index_rv[rv][i][axes[i]])
                        for i in range(len(axes))
                        if axes[i] != "None"
                    )
                final_bucket[key2].sort(key=_sort_key)

                #If all is None then we need axis_size 
                need_axis_size = all(el == "None" for el in axes)
                if(need_axis_size):
                    axis_size = len(final_bucket[key2])
                def _deserialize_index_list(blob):
                    # Inverse of _serialize_index_list.
                    # Layout: [ndim: int64][shape dims: int64*ndim][data bytes]
                    # Guard: b''.split(b'|') yields [b''] so skip empty chunks.
                    result = []
                    for part in blob.split(b"|"):
                        if not part:
                            continue
                        ndim  = int(np.frombuffer(part[:8], dtype=np.int64)[0])
                        shape = tuple(np.frombuffer(part[8:8+8*ndim], dtype=np.int64).tolist())
                        data  = np.frombuffer(part[8+8*ndim:], dtype=np.int64).reshape(shape)
                        result.append(data)
                    return result

                #Getting the remaining index pattern after removing the batching axes
                for i in range(len(key2)//2, len(key2)):
                    if(isinstance(key2[i], str)):
                        remain.append(key2[i])
                    else:
                        remain.append(_deserialize_index_list(key2[i]))
                #Looping through the parents of the RVs         
                for i in range(len(key2)//2):
                    #Get the dimension of the parent that we are batching on
                    ndim = len(index_rv[final_bucket[key2][0]][i])
                    idd = 0
                    #If the parent is not an index then we just add the parent RV 
                    # as an argument to the new vmap RV
                    if(isinstance(remain[i], str) and remain[i] == "NotIndex"):
                        c_rv.append("NotIndex")
                        continue
                    #c_rv is to store the RVs that we will use as 
                    # arguments for the new vmap RV that we are creating.
                    c_rv.append([])
                    #Loop through the dimensions of the parent and 
                    # get the indexing pattern for each dimension. 
                    # If we are batching on that dimension then we need 
                    # to create an index pattern that gets all the indexes of all RVs 
                    # on parent i on dimension j. If we are not batching on that 
                    # dimension then we can just use the remaining index pattern 
                    for j in range(ndim):
                        arr = []
                        if(j == axes[i]):
                            for rv in final_bucket[key2]:
                                arr.append(index_rv[rv][i][j])
                        else:
                            arr = remain[i][idd]
                            idd+=1
                        #This is the RV that we will use as an argument for the new vmap
                        new_rv = RV(Constant(arr))
                        c_rv[-1].append(new_rv)
                # print(key2)
                new_p = []
                #Looping through all parents. Now we start creating the Index RVs
                for i in range(len(key2)//2):
                    args = []
                    #If the parent is not an index then we just add the parent RV as 
                    # an argument to the new vmap RV
                    if(isinstance(c_rv[i], str)):
                        new_p.append(key[i+1])
                    else:
                        args.append(Index())
                        args.append(key[i+1])
                        for var in c_rv[i]:
                            args.append(var)
                        new_p.append(RV(*args))
                    axes[i] = None if axes[i] == "None" else axes[i]
                    if(new_p[i].ndim == 1):
                        axes[i] = 0 if axes[i] != None else None
                if(need_axis_size):
                    op = VMap(key[0], in_axes = tuple(axes), axis_size = axis_size)
                else:
                    op = VMap(key[0], in_axes = tuple(axes))
                final_args = [op]
                for x in new_p:
                    final_args.append(x)
                vmap = RV(*final_args)
                print(f"Created vmap: {vmap}")

                # Map each original RV to an index into vmap along its leading axis.
                # vmap.shape == (N, d1, d2, ...) where (d1,...) == original_rv.shape.
                # Index requires exactly ndim indices; output shape = concat of index shapes.
                # A scalar index (shape ()) contributes nothing; a 1-D range of length d
                # contributes (d,). So index(vmap, i, range(d1),...) -> shape (d1,...). 
                for i, original_rv in enumerate(final_bucket[key2]):
                    index_args = [Index(), vmap, RV(Constant(i))]
                    for dim_size in vmap.shape[1:]:  # no-op for 1-D vmaps
                        index_args.append(RV(Constant(list(range(dim_size)))))
                    M[original_rv] = RV(*index_args)

        return M

    def run_to_fixpoint(self, RVs):
        """
        Repeatedly apply run_all_vmaps until no new vmap opportunities remain.

        A single call to run_all_vmaps may fuse N scalar RVs into one vmap.
        But if there were M such groups, we now have M vmap nodes that share
        the same op/parent structure and may themselves be fuseable — this pass
        repeats until that process stabilises.

        Composition across passes is done with substitute_parents: if pass 1
        gives  orig → Index(vmap1, i)  and pass 2 fuses vmap1 nodes so that
        vmap1 → Index(vmap2, j),  then orig ends up as Index(Index(vmap2,j),i),
        which correctly recovers the original element from the doubly-batched result.

        Returns a single global_M mapping every original RV to its final form.
        """
        global_M = self.run_all_vmaps(RVs)

        while True:
            # --- collect frontier: vmap nodes produced in the last pass ----
            # These are the base nodes (after stripping any Index wrapper) that
            # appear in global_M values but are not yet keys in global_M,
            # meaning they are freshly created and haven't been examined yet.
            frontier = set()
            for replacement in global_M.values():
                node = replacement
                while node.op.name == "Index":
                    node = node.parents[0]
                if node not in global_M:
                    frontier.add(node)

            if not frontier:
                break

            # --- try to vmap the frontier nodes amongst themselves -----------
            M_next = self.run_all_vmaps(list(frontier))

            # Convergence: nothing in the frontier got batched
            if not any(M_next.get(rv, rv) is not rv for rv in frontier):
                # Absorb so future frontier scans don't re-examine these nodes
                global_M.update(M_next)
                break

            # --- compose: rewrite every value in global_M through M_next ----
            # substitute_parents recurses into Index chains, so
            # Index(vmap1, i) becomes Index(M_next[vmap1], i) automatically.
            for orig in list(global_M.keys()):
                global_M[orig] = self.substitute_parents(global_M[orig], M_next)

            # Absorb M_next so future iterations can trace through these nodes
            global_M.update(M_next)

        return global_M

    def substitute_parents(self, rv, M):
        """
        Return a version of rv with all parents recursively remapped through M.
        Because RVs are frozen, any substitution produces a new RV.
        Index nodes are not ranked (and therefore not keys in M), so we recurse
        through them to remap their own parents before rebuilding the chain.
        """
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
        """
        Run vmap level by level over the DAG.

        level_ranking returns lists of RVs in topological order.  For each
        level we:
          1. Rebuild every RV in the level with parents already remapped by M
             (necessary because frozen RVs cannot be mutated in place).
          2. Run run_vmap on those substituted RVs so the engine sees the
             updated parent structure and can discover new batching
             opportunities that only exist after earlier levels were fused.
          3. Merge the level's local mapping back into the global M so that
             later levels pick it up automatically via substitute_parents.

        Returns M, a dict mapping every original RV to its final replacement.
        """
        order_bucket = self.level_ranking(RVs)
        M = {}  # original RV -> final replacement RV

        for group in order_bucket:
            # --- Step 1: rebuild each RV with substituted parents ---
            # Track substituted RV -> original RV so we can write back into M
            sub_to_orig = {}
            substituted_group = []

            for rv in group:
                sub_rv = self.substitute_parents(rv, M)
                sub_to_orig[sub_rv] = rv
                substituted_group.append(sub_rv)

            # --- Step 2: attempt vmap across the substituted group ---
            group_M = self.run_vmap(substituted_group)

            # --- Step 3: merge local results into the global mapping ---
            # group_M maps substituted_rv -> final_rv (Index into vmap, or self).
            # We want M[original_rv] -> final_rv.
            for sub_rv, orig_rv in sub_to_orig.items():
                M[orig_rv] = group_M.get(sub_rv, sub_rv)
            

        return M