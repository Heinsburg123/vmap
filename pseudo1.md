# VmapEngine — High-Level Pseudocode

---

## Entry Point: run_to_fixpoint(RVs)

    RVs ← all upstream nodes of the input RVs (close under parents)
    global_M ← run_all_vmaps(RVs)

    repeat:
        frontier ← { vmap_node : vmap_node is the root of some value in global_M
                                  and vmap_node is not yet a key in global_M }
        # i.e. newly created VMap RVs that themselves may be batchable

        if frontier is empty:
            break

        M_next ← run_all_vmaps(frontier)

        if M_next made no progress (every rv in frontier maps to itself):
            merge M_next into global_M
            break

        # propagate new substitutions into existing mappings, then absorb
        for every (orig → replacement) in global_M:
            global_M[orig] ← substitute_parents(replacement, M_next)
        merge M_next into global_M

    return global_M

---

## run_all_vmaps(RVs)

    levels ← level_ranking(RVs)
    # levels is a list of groups; Index nodes are transparent/skipped throughout

    M ← batch_constants(RVs)
    # stack identical-shape Constant RVs into a single array and replace each
    # with an Index into that stacked constant

    for each level_group in levels (topological order):
        substituted ← [ substitute_parents(rv, M)  for rv in level_group ]
        # rewrite each RV's parents using the replacements found so far

        group_M ← run_vmap(substituted)
        # find and create vmaps within this level

        merge group_M into M (mapping original RVs through the substitution)

    return M

---

## run_vmap(RVs)

    # --- Phase 1: partition RVs by structural identity ---
    for each rv in RVs (skip Constants and Index nodes):
        hash_key ← compute_hash(rv)
            # = (op, resolved_parent_0, ..., resolved_parent_k, shape)
            # resolved_parent strips away any Index wrappers to the underlying RV
        add rv to hash_map[hash_key]

    # --- Phase 2: enumerate axis candidates and build bucket ---
    for each partition (hash_key → rv_list):
        bucket ← {}
        for each rv in rv_list:
            index_lst ← [ group_index(p)  for p in rv.parents ]
            # group_index returns the list of constant index arrays for parent p
            # or "NotIndex" if p is not a constant-indexed RV

            for each valid axis combination axes[] over the parents:
                # for each parent i, axes[i] is either:
                #   "None"  — this parent is not batched
                #   j       — batch along dimension j of parent i's index list
                # only scalar (0-d) index entries are candidates for batching
                remaining ← serialize( index_lst with the chosen axis entry removed per parent )
                bucket_key ← (axes[0], ..., axes[k], remaining[0], ..., remaining[k])
                add rv to bucket[bucket_key]

        # --- Phase 3: greedy set cover to select vmap groups ---
        selected ← run_greedy_set(rv_list, bucket)
        # returns a dict: bucket_key → subset of rv_list covered by that key
        # greedily maximises coverage; ties broken by most "None" axes (prefer no-batch)

        # --- Phase 4: construct each vmap ---
        for each (bucket_key → group) in selected:
            if |group| == 1:
                map the single RV to itself (no batching needed)
                continue

            sort group by _sort_key:
                # for each non-None axis i, read the scalar index value rv uses
                # at dimension axes[i] of parent i; sort lexicographically
                # so that vmap output index 0,1,2,... matches those values

            deserialize remaining index arrays from bucket_key

            for each parent i:
                reconstruct the stacked index array for parent i:
                    for each dimension j of parent i:
                        if j == axes[i]:   stack the scalar indices across the group (the batch dim)
                        else:              use the shared remaining index array for dim j

                if the resulting index arrays are just arange over all dims (no-op index):
                    new_parent[i] ← resolved_parent directly
                else:
                    new_parent[i] ← Index(resolved_parent, stacked_index_arrays...)

            compute tensor_axes[i]:
                # the actual axis number in the tensor passed to VMap,
                # accounting for which index dimensions are 1-D arrays
                # (each 1-D array is one tensor dimension)

            if all axes are "None":
                op ← VMap(hash_key.op, in_axes=tensor_axes, axis_size=|group|)
                # axis_size needed because no existing tensor dimension carries the batch
            else:
                op ← VMap(hash_key.op, in_axes=tensor_axes)

            vmap_rv ← RV(op, new_parent[0], ..., new_parent[k])

            for i, original_rv in enumerate(group):
                M[original_rv] ← Index(vmap_rv, i, arange(dim)...)
                # slice out row i from the vmap result

    return M

---

## level_ranking(RVs)

    Build adjacency list of the DAG, treating Index nodes as transparent
    (i.e. an edge A → Index(...) → B is treated as a direct edge A → B)

    Compute in-degree for each non-Index RV, counting only internal parents

    BFS (Kahn's algorithm):
        assign rank[rv] = max(rank[parent] + 1) over all resolved parents
        Constants and parentless RVs start at rank 0

    return RVs grouped and sorted by rank

---

## batch_constants(RVs)

    Group all Constant RVs by their shape
    For each shape with more than one Constant:
        stack their values into a single new Constant array
        replace each original Constant rv_i with Index(stacked, i, arange(dim)...)
    return the replacement map M

---

## group_index(rv)

    if rv is not an Index node:
        return "NotIndex"
    if any index argument cannot be resolved to a constant:
        return "NotIndex"  # random or unknown index

    own ← list of constant index arrays from rv's index arguments

    if rv's parent is itself an Index node:
        inner_lst ← group_index(rv.parent)
        if inner_lst == "NotIndex": return "NotIndex"
        merge own into inner_lst:
            # replace each "full-slice placeholder" in inner_lst with the next entry of own
        return merged list
    else:
        return own

---

## substitute_parents(rv, M)

    if rv in M: return M[rv]
    recursively substitute each parent, rebuild RV only if any parent changed

---

## run_greedy_set(universe, sets)

    Greedy set cover with lazy-deletion max-heap:
        heap key = (-coverage_size, -none_count_in_axes, insertion_counter)
        # primary: pick the key covering the most uncovered RVs
        # tiebreak: prefer keys with more "None" entries in the axes
        # tiebreak again: insertion order (stable)

    repeat until all RVs covered or heap empty:
        pop best (lazily recompute real coverage if stale)
        record selected key and its contribution
        remove covered RVs from uncovered set

    return { key → set_of_covered_RVs }
