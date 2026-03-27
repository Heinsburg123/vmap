from itertools import product
import numpy as np
import heapq
from pangolin.ir import * 

class VmapEngine:
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
        # print(ans)
        remaining = []
        for comb in ans:
            less = []
            # print(index_lst)
            for i,idx in enumerate(comb):
                if(idx == "None"):
                    if(isinstance(index_lst[i], str)):
                        less.append(index_lst[i])
                    else:
                        less.append(np.array(index_lst[i]).tobytes())
                else:
                    less.append(np.array(index_lst[i][:idx] + index_lst[i][idx+1:]).tobytes())
            remaining.append(less)
        return ans, remaining
        

    # def return_indexes(self, RVs, index_list):
    #     ans = []
    #     exception = []
    #     M = {}
    #     for i in range(len(RVs)):
    #         rv = RVs[i]
    #         dp = (rv, index_list[i])
    #         if(rv.get_shape() not in M):
    #             M[rv.get_shape()] = []
    #         M[rv.get_shape()].append(dp)
    #     for shape in M:
    #         N = {}
    #         for dp in M[shape]:
    #             if dp[1] == "NotIndex":
    #                 ans.append(["NotIndex"])
    #                 break
    #             if len(dp[1][0])!=1:
    #                 exception.append(dp[0])
    #             tp = []
    #             for i in range(1, len(dp[1])):
    #                 tp.append(tuple(dp[1][i]))
    #             tp = tuple(tp)
    #             if(tp not in N):
    #                 N[tp] = []
    #             N[tp].append(dp[0])
    #         for key in N:
    #             ans.append(N[key])
    #     return ans
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
        for rv in RVs:
            if(rv.op.name == "Constant" or rv.op.name == "Index"):
                continue
            hash_key = self.compute_hash(rv)
            if hash_key not in hash_map:
                hash_map[hash_key] = []
            hash_map[hash_key].append(rv)
        for key in hash_map:
            bucket = {}
            # print(f"Group:{key}")
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
                axes = []
                remain = []
                c_rv = []
                axis_size = 0
                for i in range(len(key2)//2):
                    axes.append(key2[i])
                need_axis_size = all(el == "None" for el in axes)
                if(need_axis_size):
                    axis_size = len(final_bucket[key2])
                for i in range(len(key2)//2, len(key2)):
                    if(isinstance(key2[i], str)):
                        remain.append(key2[i])
                    else:
                        remain.append(np.frombuffer(key2[i]))
                for i in range(len(key2)//2):
                    ndim = len(index_rv[final_bucket[key2][0]][i])
                    idd = 0
                    c_rv.append([])
                    if(remain[i] == "NotIndex"):
                        c[i].append("NotIndex")
                        continue
                    for j in range(ndim):
                        arr = []
                        if(j == axes[i]):
                            for rv in final_bucket[key2]:
                                arr.append(index_rv[rv][i][j])
                        else:
                            arr = remain[i][idd]
                            idd+=1
                        new_rv = RV(Constant(arr))
                        c_rv[-1].append(new_rv)
                new_p = []
                for i in range(len(key2)//2):
                    args = []
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
                print(vmap)

        # for key, group in hash_map.items():
        #     print(f"Group: {key}, RVs: {[rv._n for rv in group]}")
                
        
        
            