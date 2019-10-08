import sys,os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import networkx as nx


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



label_count = 0
ordinal_map = {}
vertices,edges = set(),set()

for u,v in G.edges():
    for i in (u,v):
        if i not in ordinal_map:
            ordinal_map[i] = label_count
            label_count += 1
    u, v = ordinal_map[u], ordinal_map[v]
    vertices.update({u,v})
    edges.update( {(u,v),(v,u)} )

adj_list = {}
for u,v in edges:
    if u in adj_list:
        adj_list[u].append(v)
    else:
        adj_list[u] = [ v ]
        
adj_matrix = np.array([[(1 if (i,j) in edges else 0) for i in range(len(ordinal_map))] for j in range(len(ordinal_map))])



diag      = np.diag(adj_matrix.sum(axis=0))
laplacian = diag - adj_matrix

diag_root = np.diag(adj_matrix.sum(axis=0)**-0.5)

norm_laplacian = np.dot(np.dot(diag_root,laplacian),diag_root)


eigvals, eigvecs = np.linalg.eigh(norm_laplacian)
eigvecs = eigvecs.T


sorted_eig = sorted((x,y) for x,y in zip(eigvals,eigvecs))

second_eigen_value  = sorted_eig[1][0]
second_eigen_vector = sorted_eig[1][1]


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(second_eigen_vector.reshape(-1,1))
kmeans.cluster_centers_

# sev = second_eigen_vector
# thres = np.array(sorted(sev)[:len(sev)//2]).mean()
thres = np.mean(kmeans.cluster_centers_)



rev_map = {}
for k in ordinal_map:
    rev_map[ ordinal_map[k] ] = k
    
group_1 = [rev_map[x] for x in vertices if second_eigen_vector[x]>=thres]
group_2 = [rev_map[x] for x in vertices if second_eigen_vector[x]<thres]

# pos = nx.spring_layout(G)
# pos = nx.fruchterman_reingold_layout(G)
# pos = nx.random_layout(G)
# pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.bipartite_layout(G,nodes = group_1)
pos = nx.kamada_kawai_layout(G)


names = ordinal_map
edge_freq = [(rev_map[i],rev_map[j]) for i in range(len(ordinal_map)) for j in range(len(ordinal_map)) if adj_matrix[i][j]]

edg_1 = [(i,j) for i,j in edge_freq if (i in group_1 and j in group_1)]
edg_2 = [(i,j) for i,j in edge_freq if (i in group_2 and j in group_2)]
edg_c = set(tuple(map(str,(i,j))) for i,j in edge_freq)-(set(edg_1)|set(edg_2))


plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
_ = nx.draw_networkx_nodes  (G,pos,group_1,node_color='r',node_size=50,alpha=0.5)
_ = nx.draw_networkx_nodes  (G,pos,group_2,node_color='b',node_size=50,alpha=0.5)

_ = nx.draw_networkx_edges(G,pos,edgelist=edg_1,width=3,alpha=0.1,edge_color='r')
_ = nx.draw_networkx_edges(G,pos,edgelist=edg_2,width=3,alpha=0.1,edge_color='b')
# _ = nx.draw_networkx_edges(G,pos,edgelist=edg_c,width=8,alpha=0.6,edge_color='g')
_ = nx.draw_networkx_edges(G,pos,width=1.0,alpha=1)

_ = nx.draw_networkx_labels (G,pos,font_size=8,font_color='y')
plt.axis('off')
plt.savefig('output_plots/problem_1.png')
plt.clf() # display
plt.show()
