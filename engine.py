from itertools import product
import numpy as np

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

    def run_vmap(self, RVs):
        hash_map = {}
        for rv in RVs:
            if(rv.op.name == "Constant" or rv.op.name == "Index"):
                continue
            hash_key = self.compute_hash(rv)
            if hash_key not in hash_map:
                hash_map[hash_key] = []
            hash_map[hash_key].append(rv)
            print(hash_key) 

        for key in hash_map:
            bucket = {}
            # print(f"Group:{key}")
            for rv in hash_map[key]:
                index = []
                for p in rv.parents:
                    index.append(self.group_index(p))
                get_axes, remaining = self.deep_hash(rv, index)
                for i in range(len(get_axes)):
                    axes = get_axes[i]
                    rmd = remaining[i]
                    tmp = tuple(axes+rmd)
                    if(tmp not in bucket):
                        bucket[tmp] = []
                    bucket[tmp].append(rv)
            for key2 in bucket:
                print(f"Group:{key2}, RVs:{[rv._n for rv in bucket[key2]]}")


        # for key, group in hash_map.items():
        #     print(f"Group: {key}, RVs: {[rv._n for rv in group]}")
                
        
        
            