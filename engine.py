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
        remaining = []
        for comb in ans:
            less = []
            #This is trying to get the remaining index pattern after 
            # removing the axis that we are batching on
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
        #This part is going through all the most general hash that is the Ops and parents
        for key in hash_map:
            bucket = {}
            for rv in hash_map[key]:
                index = []
                #Here is when we are going through the parents of the RV and getting 
                # the index pattern for each parent and storing it in index  
                for p in rv.parents:
                    index.append(self.group_index(p))
                #The index_rv dictionary helps us later to try and get the indexing pattern
                #for each RV 
                index_rv[rv] = index
                #Now we have the index pattern for each parent of the RV and the remaining
                # index pattern if we were to batch on a certain axis. Now 
                # we want to group them by the possible axes that we can batch on.
                get_axes, remaining = self.deep_hash(rv, index)
                #Going through all the possible axes or hash 
                for i in range(len(get_axes)):
                    axes = get_axes[i]
                    rmd = remaining[i]
                    #tmp is now the key for the bucket that we are grouping the RVs into.
                    #It consists of the axes that we are batching on and the remaining 
                    # index pattern after removing those axes
                    tmp = tuple(axes+rmd)
                    if(tmp not in bucket):
                        bucket[tmp] = []
                    bucket[tmp].append(rv)
            #Run greedy set cover on the buckets to find the best grouping 
            # of RVs to batch together
            final_bucket = self.run_greedy_set(hash_map[key], bucket)
            for key2 in final_bucket:
                final_bucket[key2] = list(final_bucket[key2])
            for key2 in final_bucket:
                axes = []
                remain = []
                c_rv = []
                axis_size = 0
                #Get the axes that we are batching on 
                for i in range(len(key2)//2):
                    axes.append(key2[i])
                #If all is None then we need axis_size 
                need_axis_size = all(el == "None" for el in axes)
                if(need_axis_size):
                    axis_size = len(final_bucket[key2])
                #Getting the remaining index pattern after removing the batching axes
                for i in range(len(key2)//2, len(key2)):
                    if(isinstance(key2[i], str)):
                        remain.append(key2[i])
                    else:
                        remain.append(np.frombuffer(key2[i]))
                #Looping through the parents of the RVs         
                for i in range(len(key2)//2):
                    #Get the dimension of the parent that we are batching on
                    ndim = len(index_rv[final_bucket[key2][0]][i])
                    idd = 0
                    #If the parent is not an index then we just add the parent RV 
                    # as an argument to the new vmap RV
                    if(isinstance(remain[i], str) and remain[i] == "NotIndex"):
                        c_rv[i].append("NotIndex")
                        continue
                    #c_rv is to store the RVs that we will use as 
                    # arguments for the new vmap RV that we are creating.
                    c_rv.append([])
                    #Loop through the dimensions of the parent and 
                    # get the indexing pattern for each dimension. 
                    # If we are batching on that dimension then we need 
                    # to create an index pattern that gets all the indexes of all RVs 
                    # on parent i on dimension j. If we are not batching on that 
                    # dimension then we can just use the remaining index pattern 
                    for j in range(ndim):
                        arr = []
                        if(j == axes[i]):
                            for rv in final_bucket[key2]:
                                arr.append(index_rv[rv][i][j])
                        else:
                            arr = remain[i][idd]
                            idd+=1
                        #This is the RV that we will use as an argument for the new vmap
                        new_rv = RV(Constant(arr))
                        c_rv[-1].append(new_rv)
                new_p = []
                #Looping through all parents. Now we start creating the Index RVs
                for i in range(len(key2)//2):
                    args = []
                    #If the parent is not an index then we just add the parent RV as 
                    # an argument to the new vmap RV
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
                print(f"Created vmap: {vmap}")

        # for key, group in hash_map.items():
        #     print(f"Group: {key}, RVs: {[rv._n for rv in group]}")
                
        
        
            