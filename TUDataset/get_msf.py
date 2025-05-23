import torch
from torch_geometric.data import DataLoader
import numpy as np
import itertools
import sys


from sklearn.cluster import SpectralClustering
from sklearn.utils import check_array
from sklearn.cluster import mean_shift
import itertools

max1 = sys.maxsize
 

import numpy as np
from sklearn.cluster import SpectralClustering

global_hash_dict = {}
       
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

# if args.dataset=='MUTAG':
#     max_num = 28
# if args.dataset=='PROTEIN':   
#     max_num = 620
# if args.dataset=='PTC_MR':
#     max_num = 28
# if args.dataset=='NCI1':   
#     max_num = 620
# if args.dataset=='NCI1':   
#     max_num = 620

max_num = 28

def fmb(i,d_m,vertices_number):
    co = torch.zeros((vertices_number, max_num))
    l1 = [ii for ii in range(vertices_number)]
    iter = itertools.combinations(l1,i)
    l = list(iter)
    l = np.array(l)
    for ii in range(len(l)):
        l2=list(l[ii])
        dm_l=d_m[:,l2]
        unique_rows = np.unique(dm_l, axis=0)
        co[l2,len(unique_rows)]=co[l2,len(unique_rows)]+1
    return co


def get_msf(edge_index1,num_nodes1,i):
    d_m1=distance_m(edge_index1,num_nodes1)
    co1=fmb(i,d_m1,num_nodes1)
    return co1






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


    

