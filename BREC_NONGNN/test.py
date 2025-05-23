import numpy as np
import argparse
from tqdm import tqdm
import logging
import itertools
import time



def func_None():
    raise NotImplementedError(f"Cannot find func {args.method}")



part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
    "CFI": (260, 360),
    # "Reliability": (400, 800),
}


parser = argparse.ArgumentParser(description="Test non-GNN methods on BREC.")
parser.add_argument("--file", type=str, default="brec_nonGNN.npy")
args = parser.parse_args()




def count_distinguish_num(graph_tuple_list):
    

    for part_name, part_range in part_dict.items():
        logging.info(f"{part_name} part starting ---")
        name=part_name
        total=0
        k=2 #k in k-GMI
        print('The order of k-msi is',k)
 
        start_time = time.time() 
        for id in tqdm(range(part_range[0], part_range[1])):
            eda=[]
            edb=[]
            a=[]
            b=[]
            pp=[]
            qq=[]
            graph_tuple = graph_tuple_list[id]
            a=graph_tuple[0]
            b=graph_tuple[1]
            qq=a.edges
            pp=b.edges

            for edge in qq:
                eda.append(edge)
            for edge in pp:
                edb.append(edge)
            
            my_array1 = np.asarray(eda)
            my_array2 = np.asarray(edb)
            a_transposed = np.transpose(my_array1)
            b_transposed = np.transpose(my_array2)
            row1_copy = np.copy(a_transposed[0])
            row2_extended = np.concatenate((a_transposed[1], row1_copy))
            row3_copy = np.copy(b_transposed[0])
            row4_extended = np.concatenate((b_transposed[1], row3_copy))

            row2_copy = np.copy(a_transposed[1])
            row1_extended = np.concatenate((a_transposed[0], row2_copy))
            row4_copy = np.copy(b_transposed[1])
            row3_extended = np.concatenate((b_transposed[0], row4_copy))

            result_arraya = np.array([row1_extended, row2_extended])
            result_arrayb = np.array([row3_extended, row4_extended])
            ma=max(result_arraya[0])+1
            mb=max(result_arrayb[0])+1
            fd,vn=get_metric_indicator(result_arraya, ma,result_arrayb, mb,k)
            if fd == 1:
                total += 1
        print('The correct number of',name,'is',total)  
        end_time = time.time()  
        execution_time = end_time - start_time  
        print("The time consumption is:", execution_time) 
         
    print('done')
        
def distance_m(edge_index,num_nodes):
    row = edge_index[0]
    col = edge_index[1]
    num_nodes = num_nodes
    am = np.zeros((num_nodes,num_nodes))
    for i in range(len(row)):
        am[row[i]][col[i]] = 1
    vertices_number = len(am)
    ls = np.eye(vertices_number)
    am[am==0] = -1
    adjacency_matrix=am+ls
    inf=float('inf')
    adjacency_matrix[adjacency_matrix==-1]=inf
    d_m=np.zeros((vertices_number,vertices_number))
    for i in range(1,num_nodes+1):
        dis = Dijkstra(adjacency_matrix,i)
        for j in range(num_nodes):
            d_m[i-1][j]=dis[j]
    return d_m

def fmb(i,d_m,vertices_number):
    co = np.zeros((vertices_number+1))
    l1 = [ii for ii in range(vertices_number)]
    iter = itertools.combinations(l1,i)
    l = list(iter)
    l = np.array(l)
    for ii in range(len(l)):
        l2=list(l[ii])
        dm_l=d_m[:,l2]
        unique_rows = np.unique(dm_l, axis=0)
        co[len(unique_rows)]=co[len(unique_rows)]+1
    return co

def main():

    graph_tuple_list = np.load(args.file, allow_pickle=True)
    count_distinguish_num(graph_tuple_list)

def are_vectors_equal(v1, v2):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            return False
    return True

def get_metric_indicator(edge_index1,num_nodes1,edge_index2,num_nodes2,i):
    d_m1=distance_m(edge_index1,num_nodes1)
    d_m2=distance_m(edge_index2,num_nodes2)
    vertices_number=num_nodes2
    # i=3 #k in k-GMI
    l1 = [ii for ii in range(vertices_number)]
    iter = itertools.combinations(l1,i)
    l = list(iter)
    l = np.array(l)
    co1=fmb(i,d_m1,num_nodes1)
    co2=fmb(i,d_m2,num_nodes2)
    if are_vectors_equal(co1, co2)==False:
        return int(1),vertices_number #if isomorphic, return 1
    else:
        return int(0),vertices_number




def Dijkstra(G, start):
    start = start - 1
    inf = float('inf')
    node_num = len(G)
    visited = [0] * node_num
    dis = {node: G[start][node] for node in range(node_num)}
    parents = {node: -1 for node in range(node_num)}
    visited[start] = 1
    last_point = start

    for i in range(node_num - 1):
        min_dis = inf
        for j in range(node_num):
            if visited[j] == 0 and dis[j] < min_dis:
                min_dis = dis[j]
                last_point = j
        visited[last_point] = 1
        if i == 0:
            parents[last_point] = start + 1
        for k in range(node_num):
            if G[last_point][k] < inf and dis[k] > dis[last_point] + G[last_point][k]:
                dis[k] = dis[last_point] + G[last_point][k]
                parents[k] = last_point + 1

    return dis



if __name__ == "__main__":
    main()
