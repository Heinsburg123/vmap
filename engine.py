
class VmapEngine:
    def group_index(self, rv):
        lst = []
        if(rv.op.name != "SimpleIndex"):
            return "NotIndex"
        while(rv.op.name == "SimpleIndex"):
            lst.append(rv.parents[1])
            rv = rv.parents[0]
        lst.reverse()
        ans = []
        for idd in lst:
            if(idd.op.name == "Constant"):
                ans.append(idd.value)
            else:
                ans.append("v_"+str(idd._n))
        return ans

    def compute_hash(self, rv):
        items = []
        items.append(rv.op.name)
        for p in rv.parents:
            tmp = p
            while(tmp.op.name == "SimpleIndex"):
                tmp = tmp.parents[0]
            items.append(tmp._n)
        return tuple(items)

    def return_indexes(self, RVs, index_list):
        ans = []
        exception = []
        M = {}
        for i in range(len(RVs)):
            rv = RVs[i]
            dp = (rv, index_list[i])
            if(rv.get_shape() not in M):
                M[rv.get_shape()] = []
            M[rv.get_shape()].append(dp)
        for shape in M:
            N = {}
            for dp in M[shape]:
                if dp[1] == "NotIndex":
                    ans.append(["NotIndex"])
                    break
                if len(dp[1][0])!=1:
                    exception.append(dp[0])
                tp = []
                for i in range(1, len(dp[1])):
                    tp.append(tuple(dp[1][i]))
                tp = tuple(tp)
                if(tp not in N):
                    N[tp] = []
                N[tp].append(dp[0])
            for key in N:
                ans.append(N[key])
        return ans

    def run_vmap(self, RVs):
        hash_map = {}
        for rv in RVs:
            if(rv.op.name == "Constant" or rv.op.name == "SimpleIndex"):
                continue
            hash_key = self.compute_hash(rv)
            if hash_key not in hash_map:
                hash_map[hash_key] = []
            hash_map[hash_key].append(rv)
        
        for key in hash_map:
            axes = []
            for i in range(0, len(hash_map[key])-1):
                lst_rv = []
                lst_id = []
                for rv in hash_map[key]: 
                    lst_id.append(self.group_index(rv.parents[i]))
                    lst_rv.append(rv.parents[i])
                tmp = self.return_indexes(lst_rv, lst_id)
                #continue here
                
        for key, group in hash_map.items():
            print(f"Group: {key}, RVs: {[rv._n for rv in group]}")
                
        
        
            