# VmapEngine — Detailed Pseudocode
# Covers non-obvious mechanics left implicit in the high-level file.

---

## deep_hash — Axis Enumeration and Serialization

    for each parent i:
        if index_lst[i] == "NotIndex":
            possible_axes[i] ← ["None"]
        else:
            # only 0-d (scalar) entries can be a batch dimension
            possible_axes[i] ← [j for j,v in enumerate(index_lst[i]) if v.ndim==0]
                                + ["None"]

    for each combo in cartesian_product(*possible_axes):
        for each parent i:
            if combo[i] == "None":
                remaining[i] ← serialize(index_lst[i])         # full list
            else:
                remaining[i] ← serialize(index_lst[i] with entry combo[i] removed)
        bucket_key ← tuple(combo) + tuple(remaining)

---

## Serialization / Deserialization

    serialize(lst of numpy arrays):
        for each arr:
            header ← int64_bytes([arr.ndim] + list(arr.shape))  # 1+ndim int64s
            emit header + arr.tobytes()
        join parts with b"|"

    deserialize(blob):
        for each part in blob.split(b"|"):
            ndim  ← int64 at bytes [0:8]
            shape ← int64s at bytes [8 : 8+8*ndim]
            data  ← int64s at bytes [8+8*ndim:] reshaped to shape

---

## run_greedy_set — Heap Key and Lazy Recomputation

    heap entry: (-coverage_size, -none_count_in_axes_half, insertion_counter, key)
    # Primary:   most uncovered RVs covered
    # Tiebreak:  most None axes (prefer no-batch / cheaper groupings)
    # Tiebreak:  insertion order (determinism)

    on pop: recompute real coverage against current uncovered set
        if shrunken: re-push with updated size (lazy deletion)
        if zero:     discard
        if fresh:    accept, remove covered items from uncovered

---

## _sort_key — Canonical VMap Ordering

    Sort group so vmap output index 0,1,2,... maps to predictable RVs.

    _sort_key(rv) ← tuple(
        int(index_lst[rv][i][axes[i]])      # scalar index at the batch dim
        for i where axes[i] != "None"
    )

    # e.g. axes=[0,"None"], RVs have scalar index 2,0,1 at dim 0 of parent 0
    # sorted order: [rv_0, rv_1, rv_2]  (indices 0,1,2)
    # → M[rv_0]=Index(vmap,0,...), M[rv_1]=Index(vmap,1,...), etc.

---

## tensor_axes — Index-List Position to Tensor Axis

    After choosing batch dim a in parent i's index list, compute the
    corresponding tensor axis, since each 1-D index array occupies one
    tensor dimension while 0-D scalars do not.

    tensor_axes[i]:
        if axes[i] is None: None
        else: count of 1-D arrays in remain[i][:axes[i]]
              (each 1-D array before position a contributes one tensor dim)

    # e.g. index_lst[i] = [scalar, arange(5), scalar, arange(7)], axes[i]=2
    #   remaining (pos 2 removed) = [scalar, arange(5), arange(7)]
    #   remaining[:2] = [scalar, arange(5)] → one 1-D array → tensor_axes[i] = 1
    #   tensor shape = [5, N, 7], batch is at axis 1

    # Special case: if new_parent[i].ndim == 1 after construction,
    #               force tensor_axes[i] = 0 (or None stays None).

---

## indices_fill_parent — Skipping No-Op Index Wrappers

    if c_rvs covers all dims and each c_rv[j].value == arange(parent.shape[j]):
        use parent directly instead of Index(parent, c_rvs...)

---

## need_axis_size — All-None Axes Case

    If every axes[i] is None, no existing tensor dimension carries the batch,
    so VMap needs an explicit axis_size = |group|.
    Otherwise axis_size is inferred from the mapped tensor dimension.

---

## per-level fixpoint in run_all_vmaps

    After run_vmap(substituted_group) → group_M:

    while True:
        frontier ← vmap nodes appearing in group_M.values()
                    that are not yet keys in group_M
                    (found by stripping Index wrappers from each value)
        if empty: break

        M_next ← run_vmap(frontier)
        made_progress ← any rv where M_next[rv] ≠ rv

        # backward substitution: update existing group_M entries so they
        # point through M_next's new mappings rather than to stale nodes
        for orig in group_M:
            group_M[orig] ← substitute_parents(group_M[orig], M_next)
        merge M_next into group_M

        if not made_progress: break

    # No backward substitution into earlier levels' M entries needed:
    # the per-level frontier only creates vmaps from the current level's
    # outputs, which earlier levels never referenced.
