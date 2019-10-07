import sys,os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

import warnings
warnings.filterwarnings("ignore")

import LA,network

input_file_path = sys.argv[1]

if 'output_plots' not in os.listdir():
    os.mkdir('output_plots')

if 'output_data' not in os.listdir():
    os.mkdir('output_data')

file_out = open('output_data/output_problem1.txt','w')

LA.streams = [sys.stdout,file_out]

print = LA.my_print

G = nx.read_gml(input_file_path,label='name')

vertices,edges = set(),set()

for x in G.edges():
    vertices.update(set(map(int,x)))
    edges.add( tuple(int(y) for y in x) )
    edges.add( tuple(int(y) for y in x[::-1]) )

adj_list = {}
for u,v in edges:
    if u in adj_list:
        adj_list[u].append(v)
    else:
        adj_list[u] = [ v ]


#TASK 1
degree_dist = {}
for x,y in edges:
    if x in degree_dist:
        degree_dist[x] += 1
    else:
        degree_dist[x] =  1

#HISTOGRAM PLOT OF NODE DEGREE CENTRALITY DISTRIBUTION
BINS = max(degree_dist.values())+1
plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
plt.hist(degree_dist.values(),range=(0,BINS),bins=BINS,align='left',rwidth=0.5)
plt.xlabel('Degree Value');plt.ylabel('Frequency of Degree');plt.title('Degree Distribution')
plt.savefig('output_plots/problem_1_task_1.png')
plt.clf()
#plt.show()


#TASK 2
vertex_freq,edge_freq = network.betweeness_centrality(vertices,edges,adj_list)

print('Top 2 Nodes with highest betweeness centrality are')
print(tuple(G._node[str(x)]['name'] for x in sorted(vertex_freq,key = lambda x:vertex_freq[x],reverse=True)[:2]))

BINS = max(vertex_freq.values())+1
plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
plt.hist(vertex_freq.values(),range=(0,BINS),bins=BINS,align='left',rwidth=0.5)
plt.xlabel('Between Centrality');plt.ylabel('Frequency of Centrality');plt.title('Vertex Centrality Distribution')
plt.savefig('output_plots/problem_1_task_2.png')
plt.clf()


#TASK 3
print('\nEdge with highest betweeness centrality is')
print(tuple(G._node[str(x)]['name'] for x in sorted(edge_freq,key = lambda x:edge_freq[x],reverse=True)[0]))


#TASK 4
#D is Degree Matrix, A is Adjacency Matrix, L is Laplacian matrix
L = [[ degree_dist[i] if i==j else (-1 if i in adj_list[j] else 0)
                         for j in sorted(vertices)]for i in sorted(vertices)]
norm_L = [[((1 if degree_dist[i] else 0) if i==j else (-1/((degree_dist[i])*(degree_dist[j]))**0.5 if i in adj_list[j] else 0))
                         for j in sorted(vertices)]for i in sorted(vertices)]
_ = LA.eigj(L)
eig = { _[0][i]:_[1].T[i]  for i in range(len(_[0]))}
if (np.array(L)==np.array(L).T).all():
    print('\nObtained Eigen Values are Real')
else:
    print('\nObtained Eigen Values are Imaginary')


#TASK 5
print('\nSmallest Eigen Value is',min(eig))
print('\nEigen Vector corresponding to Smallest Eigen value is')
print(eig[min(eig)]/eig[min(eig)].min())
print('\nDifference from Ideal Eigen value is ',min(eig))
print('\nMaximum Difference Component-wise from Ideal Eigen vector is',
	(np.array([1]*len(vertices))-(eig[min(eig)]/eig[min(eig)].min())).max()
	)
_ = min(eig),eig[min(eig)]
del eig[min(eig)]
print('\nSecond Smallest Eigen Value is',min(eig))
print('\nEigen Vector corresponding to Second Smallest Eigen value is')
print(eig[min(eig)])
eig[_[0]] = _[1]


#Task 6
second_smallest     = sorted(eig)[1]
second_eigen_vector = eig[second_smallest]

group_1 = [x for x in vertices if second_eigen_vector[x-1]>=0]
group_2 = [x for x in vertices if second_eigen_vector[x-1]<0]

lys = ['fruchterman_reingold_layout',
       'kamada_kawai_layout','spring_layout']

pos = nx.kamada_kawai_layout(G)
names = {x:G.nodes._nodes[x]['name'] for x in G.nodes._nodes}
edg_1 = [tuple(map(str,(i,j))) for i,j in edge_freq if (i in group_1 and j in group_1)]
edg_2 = [tuple(map(str,(i,j))) for i,j in edge_freq if (i in group_2 and j in group_2)]
edg_c = set(tuple(map(str,(i,j))) for i,j in edge_freq)-(set(edg_1)|set(edg_2))

plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
_ = nx.draw_networkx_nodes  (G,pos,tuple(map(str,group_1)),node_color='r',node_size=500,alpha=0.8)
_ = nx.draw_networkx_nodes  (G,pos,tuple(map(str,group_2)),node_color='b',node_size=500,alpha=0.8)

_ = nx.draw_networkx_edges(G,pos,edgelist=edg_1,width=6,alpha=0.3,edge_color='r')
_ = nx.draw_networkx_edges(G,pos,edgelist=edg_2,width=6,alpha=0.3,edge_color='b')
_ = nx.draw_networkx_edges(G,pos,edgelist=edg_c,width=8,alpha=0.6,edge_color='g')
_ = nx.draw_networkx_edges(G,pos,width=2.0,alpha=1)

_ = nx.draw_networkx_labels (G,pos,names,font_size=14)
plt.axis('off')
plt.savefig('output_plots/problem_1_task_6.png')
plt.clf() # display


#Task 7
degree_dist_1 = [degree_dist[i] for i in group_1]
degree_dist_2 = [degree_dist[i] for i in group_2]
var_1 = np.var(degree_dist_1)
var_2 = np.var(degree_dist_2)
if var_1>=var_2:
    print('\nAlgebricians should be friend with ',{names[str(x)] for x in group_1})
else:
    print('\nAlgebricians should be friend with ',{names[str(x)] for x in group_2})


#BONUS 1
eigv,eigV = np.linalg.eigh(L)
eigV = eigV.T
partition = eigV[list(eigv).index(sorted(eigv)[1])] #CHKPNT
group_1_2 = [x for x in vertices if partition[x-1]>=0]
group_2_2 = [x for x in vertices if partition[x-1]<0]
if group_1 in (group_1_2,group_2_2):
    print("\nSame Clustering using Numpy's & Our's Second Smallest Eigen value's corresponding Eigen vector is obtained")
else:
    print('\nClustering using NumPy is bet different, check drawn graph')

pos = nx.kamada_kawai_layout(G)
edg_1 = [tuple(map(str,(i,j))) for i,j in edge_freq if (i in group_1_2 and j in group_1_2)]
edg_2 = [tuple(map(str,(i,j))) for i,j in edge_freq if (i in group_2_2 and j in group_2_2)]
edg_c = set(tuple(map(str,(i,j))) for i,j in edge_freq)-(set(edg_1)|set(edg_2))

plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
_ = nx.draw_networkx_nodes  (G,pos,tuple(map(str,group_1_2)),node_color='r',node_size=500,alpha=0.8)
_ = nx.draw_networkx_nodes  (G,pos,tuple(map(str,group_2_2)),node_color='b',node_size=500,alpha=0.8)

_ = nx.draw_networkx_edges(G,pos,edgelist=edg_1,width=6,alpha=0.3,edge_color='r')
_ = nx.draw_networkx_edges(G,pos,edgelist=edg_2,width=6,alpha=0.3,edge_color='b')
_ = nx.draw_networkx_edges(G,pos,edgelist=edg_c,width=8,alpha=0.6,edge_color='g')
_ = nx.draw_networkx_edges(G,pos,width=2.0,alpha=1)

_ = nx.draw_networkx_labels (G,pos,names,font_size=14)
plt.axis('off')
plt.savefig('output_plots/problem_1_bonus_1.png')
plt.clf()


#BONUS 2
#SEE REPORT

#BONUS 3
'Second Largest Eigen Value is taken'
partition = eigV[list(eigv).index(sorted(eigv)[-2])]
group_1_3 = [x for x in vertices if partition[x-1]>=0]
group_2_3 = [x for x in vertices if partition[x-1]<0]

pos = nx.kamada_kawai_layout(G)
plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
_ = nx.draw_networkx_nodes  (G,pos,tuple(map(str,group_1_3)),node_color='r',node_size=500,alpha=0.8)
_ = nx.draw_networkx_nodes  (G,pos,tuple(map(str,group_2_3)),node_color='b',node_size=500,alpha=0.8)

_ = nx.draw_networkx_edges(G,pos,width=2.0,alpha=1)

_ = nx.draw_networkx_labels (G,pos,names,font_size=14)
plt.axis('off')
plt.savefig('output_plots/problem_1_bonus_3.png')
plt.clf()


#BONUS 4
#SEE REPORT


file_out.close()