"""
Run test_matrix_matrix_product_3 twice (with and without None-count),
logging every round's candidate buckets and the chosen one.

This monkeypatches engine3.BucketHeap for the duration of the run, so
engine3.py and bucket_heap.py stay completely untouched.
"""

import engine3
from bucket_heap_logging import LoggingBucketHeap


def run_with_trace(build_fn, force_none_count_zero):
    """
    build_fn: a zero-arg function that builds a fresh VmapEngine and returns
              (engine, DRVs_or_targets) -- adapt to however your test
              constructs `engine` and the list passed to run_all_vmaps.
    """
    original_bucket_heap = engine3.BucketHeap

    def _factory(eng):
        return LoggingBucketHeap(eng, force_none_count_zero=force_none_count_zero,
                                  verbose=True)

    engine3.BucketHeap = _factory
    try:
        result_engine, targets = build_fn()
        M = result_engine.run_all_vmaps(targets)
        heap_trace = None
        # run_all_vmaps builds its own heap internally and doesn't return it,
        # so if you want the trace object itself afterward, either:
        #  (a) temporarily add `self.last_heap = heap` inside run_all_vmaps, or
        #  (b) capture it via a closure/list as below.
        return M
    finally:
        engine3.BucketHeap = original_bucket_heap


# ---- capture the heap object itself (so .trace is inspectable after) ----
_captured_heaps = []

def _capturing_factory(force_none_count_zero):
    def factory(eng):
        h = LoggingBucketHeap(eng, force_none_count_zero=force_none_count_zero,
                               verbose=True)
        _captured_heaps.append(h)
        return h
    return factory


def run_traced(build_fn, force_none_count_zero):
    """
    build_fn() -> (engine, targets_list)
    Returns (M, heap) where heap.trace is the full round-by-round log.
    """
    original_bucket_heap = engine3.BucketHeap
    engine3.BucketHeap = _capturing_factory(force_none_count_zero)
    try:
        eng, targets = build_fn()
        M = eng.run_all_vmaps(targets)
        heap = _captured_heaps[-1]
        return M, heap
    finally:
        engine3.BucketHeap = original_bucket_heap


if __name__ == "__main__":
    import pangolin.interface as pi  # adjust import to match your actual pi module
    from engine3 import VmapEngine

    def build_matrix_matrix_test():
        eng = VmapEngine()
        A = pi.constant([[1, 2], [3, 4]])
        b_col0 = pi.constant([5, 7])
        b_col1 = pi.constant([6, 8])

        c00 = A[0, 0] * b_col0[0] + A[0, 1] * b_col0[1]
        c10 = A[1, 0] * b_col0[0] + A[1, 1] * b_col0[1]
        c01 = A[0, 0] * b_col1[0] + A[0, 1] * b_col1[1]
        c11 = A[1, 0] * b_col1[0] + A[1, 1] * b_col1[1]

        return eng, [c00, c10, c01, c11]

    print("\n\n################ WITH None-count (real priority) ################")
    M_with, heap_with = run_traced(build_matrix_matrix_test, force_none_count_zero=False)

    print("\n\n################ WITHOUT None-count (ablation) ################")
    M_without, heap_without = run_traced(build_matrix_matrix_test, force_none_count_zero=True)

    # Diff: find the first round where the chosen group differs
    print("\n\n################ DIVERGENCE CHECK ################")
    for i, (r_with, r_without) in enumerate(zip(heap_with.trace, heap_without.trace)):
        chosen_with = r_with["chosen"]
        chosen_without = r_without["chosen"]
        if chosen_with != chosen_without:
            print(f"First divergence at round {i+1}:")
            print(f"  WITH None-count    chose: {chosen_with}")
            print(f"  WITHOUT None-count chose: {chosen_without}")
            print(f"  Candidates at that round (WITH):    {r_with['candidates']}")
            print(f"  Candidates at that round (WITHOUT): {r_without['candidates']}")
            break
    else:
        print("No divergence detected in the rounds compared "
              "(one run may have more/fewer rounds -- check lengths: "
              f"{len(heap_with.trace)} vs {len(heap_without.trace)})")