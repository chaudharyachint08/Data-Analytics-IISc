import sys,os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")


if 'output_plots' not in os.listdir():
    os.mkdir('output_plots')

if 'output_data' not in os.listdir():
    os.mkdir('output_data')


def set_part_streams(part=0):
    global streams
    file_out = open(os.path.join('output_data','{}.txt'.format(part)),'w')
    streams = [sys.stdout,file_out]

def my_print(*args,**kwargs):
    for st in streams:
        print(*args,**kwargs,file=st)


import zipfile

def extract_zip(zip_path,folder_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)



extract_zip( os.path.join('..','data','dolphins.zip') , 'temp_data')


for x in os.listdir('temp_data'):
    if x.endswith('gml'):
        G = nx.read_gml(os.path.join('temp_data',x))

partition = community_louvain.best_partition(G)
while True:
    community_labels = set(partition.values())
    if len(community_labels) <= 2:
        break

    max_val, max_indx = -1*float('inf'), (None,None)

    for i in community_labels:
        temp = deepcopy(partition)
        for j in community_labels:
            if i!=j:
                for key in partition:
                    if(partition[key]==j):
                        temp[key]=i
                val = community_louvain.modularity(temp,G)
                if val>max_val:
                    max_val = val
                    max_indx = (i,j)

    temp = deepcopy(partition)
    for key in temp:
        if partition[key] in max_indx:
            partition[key] = max_indx[0]


final_labels = tuple(set(partition.values()))

group_1 = list(filter(lambda x:partition[x]==final_labels[0],partition))
group_2 = list(filter(lambda x:partition[x]==final_labels[1],partition))

# pos = nx.spring_layout(G)
# pos = nx.fruchterman_reingold_layout(G)
# pos = nx.random_layout(G)
# pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.bipartite_layout(G,nodes = group_1)
pos = nx.kamada_kawai_layout(G)

ordinal_map = nodes=np.array(G.node)




edg_1 = [(u,v) for u,v in G.edges() if ((u in group_1) and (v in group_1))]
edg_2 = [(u,v) for u,v in G.edges() if ((u in group_2) and (v in group_2))]
edg_c = set(G.edges())-(set(edg_1)|set(edg_2))


plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
_ = nx.draw_networkx_nodes  (G,pos,group_1,node_color='r',node_size=50,alpha=0.5)
_ = nx.draw_networkx_nodes  (G,pos,group_2,node_color='b',node_size=50,alpha=0.5)


nx.draw_networkx_edges(G, pos,alpha=0.8)
_ = nx.draw_networkx_edges(G,pos,edgelist=edg_1,width=3,alpha=0.1,edge_color='r')
_ = nx.draw_networkx_edges(G,pos,edgelist=edg_2,width=3,alpha=0.1,edge_color='b')
# _ = nx.draw_networkx_edges(G,pos,edgelist=edg_c,width=8,alpha=0.6,edge_color='g')
_ = nx.draw_networkx_edges(G,pos,width=1.0,alpha=1)

_ = nx.draw_networkx_labels (G,pos,font_size=8,font_color='y')
plt.axis('off')
plt.savefig('output_plots/problem_2.png')
plt.clf() # display
plt.show()
































