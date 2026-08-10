import heapq

def _priority(hash_key, group):
    none_count = sum(
        1 for pk in hash_key[1:]
        if not isinstance(pk, tuple)
        or (len(pk) == 3 and pk[1] == "None")
    )
    none_count = 0
    return (len(group), none_count)

class BucketHeap:

    def __init__(self, engine):
        self.engine = engine
        self.buckets = {}         
        self.drv_keys = {}        
        self.dependents = {}      
        self.heap = []            
        self.key_seq = {}        
        self._seq = 0

    def _push(self, hash_key):
        group = self.buckets.get(hash_key)
        if not group:
            return
        self._seq += 1
        self.key_seq[hash_key] = self._seq
        size, none_count = _priority(hash_key, group)
        heapq.heappush(self.heap, ((-size, -none_count), self._seq, hash_key))

    def _register_dependents(self, drv):
        for p in drv.parents:
            self.dependents.setdefault(p._n, set()).add(drv)

    def _bucket_drv(self, drv):
        hash_keys = self.engine.compute_hash_keys(drv)
        self.drv_keys[drv._n] = hash_keys
        for hash_key in hash_keys:
            self.buckets.setdefault(hash_key, {})[drv._n] = drv
            self._push(hash_key)
        self._register_dependents(drv)

    def _unbucket_drv(self, drv):
        for hash_key in self.drv_keys.get(drv._n, ()):
            group = self.buckets.get(hash_key)
            if not group:
                continue
            group.pop(drv._n, None)
            if not group:
                del self.buckets[hash_key]
            else:
                self._push(hash_key)
        self.drv_keys.pop(drv._n, None)

    def build(self, DRVs):
        self.engine.index_rv = {}
        for drv in DRVs:
            self._bucket_drv(drv)

    def pop_max(self):
        while self.heap:
            neg_p, seq, hash_key = self.heap[0]
            group = self.buckets.get(hash_key)
            if not group or self.key_seq.get(hash_key) != seq:
                heapq.heappop(self.heap)   # stale entry, discard
                continue
            if len(group) <= 1:
                return None, None
            return hash_key, list(group.values())
        return None, None

    def remove_group(self, group):
        for drv in list(group):
            self._unbucket_drv(drv)

    def add_drv(self, drv):
        self._bucket_drv(drv)

    def refresh(self, drvs):
        for drv in drvs:
            self._unbucket_drv(drv)
        for drv in drvs:
            self._bucket_drv(drv)

    def dependents_of(self, drvs):
        touched = set()
        for drv in drvs:
            touched.update(self.dependents.get(drv._n, ()))
        return touched