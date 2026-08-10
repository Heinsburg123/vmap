"""
Non-invasive logging wrapper for BucketHeap.

Usage:
    from bucket_heap_logging import LoggingBucketHeap
    heap = LoggingBucketHeap(self, force_none_count_zero=False)  # or True for the ablation
    ...
    # after engine.run_all_vmaps(...) finishes, inspect heap.trace or it will
    # already have been printed round-by-round.

This does NOT modify bucket_heap.py. It subclasses BucketHeap and overrides
only the two methods needed to observe/force behavior:
  - _priority: optionally force none_count=0 (the ablation), and log what the
    real (size, none_count) would have been regardless.
  - pop_max: log every candidate bucket alive in H at the moment of selection
    (size + none_count + a short signature summary), plus which one won.

To use it in engine3.py without touching bucket_heap.py, just swap the
import / instantiation:
    heap = LoggingBucketHeap(self, force_none_count_zero=<True/False>)
instead of:
    heap = BucketHeap(self)
"""

import heapq
from bucket_heap import BucketHeap, _priority


def _summarize_key(hash_key):
    """Short human-readable summary of a hash_key: (op, is_given, *parent_keys)."""
    op = hash_key[0]
    is_given = hash_key[1]
    parts = []
    for pk in hash_key[2:]:
        if not isinstance(pk, tuple):
            parts.append(f"P#{getattr(pk, '_n', pk)}")  # literal parent object (None-axis case)
        elif len(pk) == 2 and pk[0] == "Const":
            parts.append(f"Const{pk[1]}")
        elif len(pk) == 3:
            base, ax, rest = pk
            base_id = getattr(base, "_n", base)
            parts.append(f"Idx(base=#{base_id},ax={ax},rest={rest})")
        else:
            parts.append(str(pk))
    return f"{getattr(op, 'name', op)}|given={is_given}|[{', '.join(parts)}]"


class LoggingBucketHeap(BucketHeap):
    def __init__(self, engine, force_none_count_zero=False, verbose=True):
        super().__init__(engine)
        self.force_none_count_zero = force_none_count_zero
        self.verbose = verbose
        self.round_num = 0
        self.trace = []  # list of dicts, one per round, for later inspection

    def _priority(self, hash_key, group):
        # mirrors bucket_heap._priority but lets us force none_count=0 for
        # the ablation while still recording the TRUE none_count for logging.
        true_size, true_none_count = _priority(hash_key, group)
        used_none_count = 0 if self.force_none_count_zero else true_none_count
        return (true_size, true_none_count, used_none_count)

    def _push(self, hash_key):
        group = self.buckets.get(hash_key)
        if not group:
            return
        self._seq += 1
        self.key_seq[hash_key] = self._seq
        size, true_none_count, used_none_count = self._priority(hash_key, group)
        # heap is sorted by (-size, -used_none_count) so the ablation actually
        # changes selection; true_none_count is carried along only for logging.
        heapq.heappush(
            self.heap,
            ((-size, -used_none_count), self._seq, hash_key, true_none_count),
        )

    def pop_max(self):
        # First, snapshot every *currently live* top-sized candidate for logging,
        # without disturbing the real heap state (peek-only pass).
        candidates = []
        seen_keys = set()
        for neg_p, seq, hash_key, true_none_count in list(self.heap):
            if self.key_seq.get(hash_key) != seq:
                continue  # stale, would be discarded by real pop_max too
            group = self.buckets.get(hash_key)
            if not group or len(group) <= 1:
                continue
            if hash_key in seen_keys:
                continue
            seen_keys.add(hash_key)
            candidates.append((len(group), true_none_count, hash_key, group))

        if candidates:
            max_size = max(c[0] for c in candidates)
            top = [c for c in candidates if c[0] == max_size]
            top.sort(key=lambda c: -c[1])  # best true none_count first, just for display

        # Now do the real pop using the (possibly ablated) priority.
        hash_key, group_list = super().pop_max()

        self.round_num += 1
        record = {
            "round": self.round_num,
            "num_candidates_at_max_size": len(top) if candidates else 0,
            "candidates": [
                {
                    "size": c[0],
                    "none_count": c[1],
                    "signature": _summarize_key(c[2]),
                    "members": [d._n for d in c[3].values()],
                }
                for c in (top if candidates else [])
            ],
            "chosen": None,
        }

        if hash_key is not None:
            record["chosen"] = {
                "size": len(group_list),
                "signature": _summarize_key(hash_key),
                "members": [d._n for d in group_list],
            }

        self.trace.append(record)

        if self.verbose:
            print(f"\n=== Round {self.round_num} "
                  f"(force_none_count_zero={self.force_none_count_zero}) ===")
            if candidates:
                print(f"  Candidates tied at max size={max_size}:")
                for c in top:
                    marker = "  <-- CHOSEN" if (hash_key is not None
                                                 and c[2] == hash_key) else ""
                    print(f"    size={c[0]} none_count={c[1]} "
                          f"members={[d._n for d in c[3].values()]} "
                          f"sig={_summarize_key(c[2])}{marker}")
            if hash_key is not None:
                print(f"  => Fusing group of size {len(group_list)}: "
                      f"members={[d._n for d in group_list]}")
            else:
                print("  => No group selected, halting.")

        return hash_key, group_list