# VmapEngine — High-Level Pseudocode

---

## Entry Point: run_all_vmaps(RVs)

    RVs   ← upstream_nodes(RVs)
    levels ← level_ranking(RVs)        # topological levels, Index nodes transparent
    M     ← batch_constants(RVs)       # stack same-shape Constants into one array

    for each level in levels:
        substituted ← [substitute_parents(rv, M) for rv in level]
        group_M     ← run_vmap(substituted)

        # per-level: keep find vmap until no longer able
        while True:
            frontier ← { root(v) : v in group_M.values(), root(v) not in group_M }
            # root() strips Index wrappers to reach the underlying VMap node
            if frontier is empty: break

            M_next       ← run_vmap(frontier)
            made_progress ← any rv in frontier where M_next[rv] ≠ rv

            backward-substitute all entries in group_M using M_next
            merge M_next into group_M

            if not made_progress: break

        merge group_M into M   # includes both original and frontier-level mappings

    return M

---

## run_vmap(RVs)

    # Phase 1 — partition by structural identity
    for each rv in RVs (skip Constants, Index nodes):
        hash_key ← (op, resolved_parent_0, ..., resolved_parent_k, shape)
        add rv to hash_map[hash_key]

    # Phase 2 — enumerate axis candidates, build bucket
    for each partition (hash_key → rv_list):
        for each rv in rv_list:
            index_lst[rv] ← [group_index(p) for p in rv.parents]

            for each valid axis combo (axes[0], ..., axes[k]):
                # axes[i] ∈ { scalar positions in index_lst[rv][i] } ∪ { None }
                bucket_key ← (axes..., serialize(index_lst with axis entry removed)...)
                add rv to bucket[bucket_key]

        # Phase 3 — greedy set cover; ties broken by most None axes
        selected ← run_greedy_set(rv_list, bucket)

        # Phase 4 — construct each vmap
        for each (bucket_key → group) in selected:
            if |group| == 1: M[rv] ← rv; continue

            sort group by scalar index values at each non-None axis (_sort_key)

            for each parent i:
                reconstruct index arrays:
                    batched dim  → stack scalar indices across the group
                    remaining dims → shared constant arrays from bucket_key
                new_parent[i] ← Index(resolved_parent, reconstructed arrays)
                              or resolved_parent directly if arrays are no-ops

            tensor_axes[i] ← position of batch dim in the actual tensor
                              (counts preceding 1-D index arrays in remaining)

            if all axes are None:
                op ← VMap(base_op, in_axes=tensor_axes, axis_size=|group|)
            else:
                op ← VMap(base_op, in_axes=tensor_axes)

            vmap ← RV(op, new_parent[0], ..., new_parent[k])
            for i, rv in enumerate(group):
                M[rv] ← Index(vmap, i, arange(dim)...)

    return M

---

## level_ranking(RVs)

    Build DAG adjacency list treating Index nodes as transparent edges.
    Compute in-degree per non-Index RV counting only internal parents.
    BFS (Kahn's): rank[rv] = max(rank[parent] + 1) over resolved parents.
    Return RVs grouped and sorted by rank.

---

## batch_constants(RVs)

    Group Constants by shape.
    For each shape with >1 Constant: stack into one array, replace each with Index(stacked, i, ...).
    Return replacement map.

---

## group_index(rv)  →  flat list of constant index arrays, or "NotIndex"

    Flatten nested Index chains into one list of constant index arrays
    (one per dimension of the base array), recursively merging inner and outer
    index lists.  Return "NotIndex" if any index is non-constant or non-Index.

---

## run_greedy_set(universe, sets)  →  { key → covered subset }

    Lazy-deletion max-heap.  Heap key: (-coverage, -none_count, counter).
    Each iteration: pop best key, recompute real coverage, re-push if stale,
    otherwise record and remove covered items from uncovered set.
