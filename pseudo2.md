# VmapEngine — Detailed Pseudocode
# Covers the non-obvious mechanics left implicit in the high-level file.

---

## group_index(rv) — Nested Index Merging

The goal is to flatten a chain of nested Index ops into a single flat list of
constant index arrays, one per dimension of the underlying base array.

    if rv.op != "Index":
        return "NotIndex"

    own = []
    for each index argument p of rv (skipping the first parent, which is the array):
        c = get_constant(p)       # recursively resolve to a numpy scalar/array
        if c == "Random":
            return "NotIndex"     # non-constant index; can't batch
        own.append(c)

    if rv.parents[0].op != "Index":
        return own                # base case: single level of indexing

    # Recursive case: rv = outer_index[ inner_index[base] ]
    inner_lst = group_index(rv.parents[0])
    if inner_lst == "NotIndex":
        return "NotIndex"

    # Merge own into inner_lst.
    # inner_lst describes how the intermediate array was sliced from base.
    # own describes how rv further slices the intermediate array.
    # A "full-slice placeholder" in inner_lst is a 1-D array equal to arange(n);
    # it means that dimension was left whole, so the outer index can refine it.
    # A non-arange entry in inner_lst is a concrete selection; it is kept as-is.

    merged = []
    j = 0                          # pointer into own[]
    for entry in inner_lst:
        arr = np.asarray(entry)
        is_full_slice = (arr.ndim == 1 and arr == arange(len(arr)))
        if is_full_slice and j < len(own):
            merged.append(own[j])  # replace placeholder with outer's refinement
            j += 1
        else:
            merged.append(entry)   # keep inner's concrete index unchanged
    merged += own[j:]              # any remaining own entries (extra dims)

    return merged

    # Example:
    #   base shape = [4, 5]
    #   inner: rv1 = base[arange(4), 2]   → inner_lst = [arange(4), 2]
    #   outer: rv2 = rv1[3]               → own = [3]
    #   merge: arange(4) is a full slice → replace with own[0]=3; keep 2 as-is
    #   result: [3, 2]   (i.e. rv2 == base[3, 2])

---

## get_constant(rv) — Recursive Constant Resolution

    if rv.op == "Constant":
        return rv.op.value

    if rv.op.random == True:
        return "Random"

    # rv is a deterministic op applied to parents (e.g. an Index)
    # collect all non-first-parent index arguments (in reverse, then reverse back)
    index_args = []
    for p in rv.parents[1:]:           # index arguments
        c = get_constant(p)
        if c == "Random": return "Random"
        index_args.append(c)

    base = get_resolved_parent(rv)     # strip all Index wrappers
    if base.op.random: return "Random"

    return base.op.value[tuple(index_args)]   # numpy fancy indexing

---

## deep_hash(rv, index_lst) — Axis Enumeration and Serialization

index_lst[i] is either:
  - "NotIndex"  — parent i is not constant-indexed (treat as broadcast/shared)
  - a list of numpy arrays, one per dimension of parent i

Goal: for each possible choice of which dimension to batch over per parent,
produce (1) the axes[] choice vector, (2) the "remaining" signature — a
serialized form of the index arrays with the chosen axis's entry removed.

    # Build candidate axes for each parent
    possible_axes = []
    for idd in index_lst:
        if idd == "NotIndex":
            possible_axes.append(["None"])          # can only be not-batched
        else:
            # only 0-d (scalar) index entries can be the batch dimension
            candidates = [ j for j, val in enumerate(idd) if val.ndim == 0 ]
            candidates.append("None")               # always include the no-batch option
            possible_axes.append(candidates)

    # Enumerate all combinations
    all_axis_combos = cartesian_product(*possible_axes)
    # e.g. if parent 0 has candidates [0, "None"] and parent 1 has ["None"]:
    #   combos = [(0,"None"), ("None","None")]

    # For each combo, compute the "remaining" signature
    # The remaining for parent i = the index list with the chosen axis entry removed
    # (or the full list serialized if axis is "None")
    remaining_list = []
    for axes in all_axis_combos:
        remaining = []
        for i, axis in enumerate(axes):
            if axis == "None":
                if index_lst[i] == "NotIndex":
                    remaining.append("NotIndex")
                else:
                    remaining.append( serialize(index_lst[i]) )
            else:
                # remove entry at position `axis` from the index list
                remaining.append( serialize(index_lst[i][:axis] + index_lst[i][axis+1:]) )
        remaining_list.append(remaining)

    return all_axis_combos, remaining_list

---

## Serialization / Deserialization of Index Lists

Used to make numpy array lists hashable (for use as dict keys).

    serialize(lst):
        # lst is a list of numpy arrays (the index arrays for one parent,
        # possibly with the batch dimension removed)
        parts = []
        for arr in lst:
            # header: [ndim, dim0, dim1, ..., dim_{ndim-1}] as int64 bytes
            header = int64_bytes([arr.ndim] + list(arr.shape))
            parts.append(header + arr.data_bytes)
        return b"|".join(parts)

    deserialize(blob):
        result = []
        for part in blob.split(b"|"):
            ndim  = int64_from_bytes(part[0:8])
            shape = int64_array_from_bytes(part[8 : 8 + 8*ndim])
            data  = int64_array_from_bytes(part[8 + 8*ndim :]).reshape(shape)
            result.append(data)
        return result

    # Format per array:
    #   bytes 0..7          : ndim (int64)
    #   bytes 8..8+8*ndim-1 : shape (ndim int64s)
    #   remaining bytes     : flattened int64 array data
    # Arrays within one parent separated by b"|"

---

## run_greedy_set — Heap Key and Lazy Recomputation

    # Heap entry: (-coverage_size, none_count, insertion_counter, bucket_key)
    #
    # Primary sort:   most RVs covered first (negate for min-heap)
    # Secondary sort: most "None" entries in the axes half of the key
    #                 (prefer no-batch groupings — they are cheaper and
    #                  more general, and coverage size is equal)
    # Tertiary sort:  insertion counter (FIFO tiebreak for determinism)

    none_count(key):
        axes_half = key[ : len(key)//2 ]
        return -sum(1 for el in axes_half if el == "None")
        # negative so that more Nones → smaller heap key → popped first

    # Lazy deletion: when a key is popped, recompute its real coverage
    # against the still-uncovered set.  If it has shrunk, re-push with
    # updated size rather than processing immediately.
    # This avoids rebuilding the heap after every selection.

    pop (neg_size, nc, ctr, key):
        real = |sets[key] ∩ uncovered|
        if real == 0: discard, continue
        if real < -neg_size:
            push (-real, none_count(key), new_counter, key)   # stale; re-push
            continue
        # real == -neg_size: entry is fresh, use it
        result[key] = sets[key] ∩ uncovered
        uncovered -= result[key]

---

## _sort_key — Canonical Ordering of RVs Within a Vmap Group

Before stacking a group of RVs into a vmap, we must agree on an order so that
vmap output index 0 corresponds to a predictable RV.  The sort key is the
tuple of scalar index values along each batched dimension.

    _sort_key(rv):
        return tuple(
            int( index_rv[rv][i][ axes[i] ] )      # scalar index at the batch dim
            for i in range(len(axes))
            if axes[i] != "None"
        )

    # Example:
    #   axes = [0, "None"]  (batch over dim 0 of parent 0, parent 1 is shared)
    #   rv_A has index_rv[rv_A][0] = [scalar(2), arange(5)]  → sort value = (2,)
    #   rv_B has index_rv[rv_B][0] = [scalar(0), arange(5)]  → sort value = (0,)
    #   rv_C has index_rv[rv_C][0] = [scalar(1), arange(5)]  → sort value = (1,)
    #   sorted order: rv_B, rv_C, rv_A
    #   → vmap[0] = rv_B, vmap[1] = rv_C, vmap[2] = rv_A
    #   → each original rv maps to Index(vmap, its position in sorted order)

---

## tensor_axes — Mapping Index-List Position to Tensor Axis

After choosing which entry in an index list is the batch dimension (call it `a`),
we need to know which axis of the *actual tensor* passed to VMap corresponds
to position `a`.  This differs from `a` because each 1-D index array in the
index list occupies exactly one tensor dimension, while scalars (0-D) do not.

    tensor_axes[i]:
        if axes[i] is None:
            tensor_axes[i] = None
        else:
            a = axes[i]
            # count how many entries *before* position a in the remaining list
            # are 1-D arrays (i.e., each contributes one tensor dim)
            tensor_axes[i] = count of entries e in remain[i][:a]
                             where np.asarray(e).ndim == 1

    # Example:
    #   index list for parent i = [scalar(3), arange(5), scalar(1), arange(7)]
    #   axes[i] = 2  (batch over the second scalar, at position 2)
    #   remaining (with pos 2 removed) = [scalar(3), arange(5), arange(7)]
    #   how many 1-D arrays appear before position 2 in remaining?
    #     remaining[:2] = [scalar(3), arange(5)]  → one 1-D array
    #   tensor_axes[i] = 1
    #   (the tensor passed to VMap has shape [5, N, 7] where N is the batch dim,
    #    and the batch is at tensor axis 1)

    # Special case: if the tensor has only one dimension after all substitutions
    # (new_p[i].ndim == 1), force tensor_axes[i] = 0 (or None stays None).
    # This handles the case where all non-batch index arrays collapsed away.

---

## indices_fill_parent — Detecting No-Op Index

If after reconstruction all index arrays are exactly arange over their
respective dimension, the Index wrapper is a no-op and we skip it.

    indices_fill_parent(c_rvs, parent):
        if len(c_rvs) != len(parent.shape):
            return False
        for each (idx_rv, dim_size) in zip(c_rvs, parent.shape):
            val = idx_rv.op.value
            if val.ndim != 1: return False
            if len(val) != dim_size: return False
            if val != arange(dim_size): return False
        return True
        # If True, use parent directly instead of wrapping in Index(parent, c_rvs...)

---

## need_axis_size — When All Axes Are "None"

If every entry in axes[] is "None", no parent carries an existing tensor
dimension over which to map.  The batch axis is entirely new, so VMap needs
an explicit axis_size telling it how many times to replicate the operation.

    need_axis_size = all(ax == "None" for ax in axes)
    if need_axis_size:
        op = VMap(base_op, in_axes=tensor_axes, axis_size=len(group))
    else:
        op = VMap(base_op, in_axes=tensor_axes)
        # axis_size is inferred from the size of the tensor at the mapped axis
