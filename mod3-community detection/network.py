#Finding Betweeness centrality of Vertex/Edge using BFS for shortest path
def betweeness_centrality(vertices,edges,adj_list):
	#Global collection of V & E freq in shortest paths 
	vertex_freq,edge_freq = { v:0 for v in vertices },{ e:0 for e in edges }
	def BFS(src,trgt):
	    marked[trgt] = False
	    nxt = {x for x in adj_list[trgt] if marked[x]}
	    tmp_vertex_freq[trgt] += 1
	    if nxt:
	        for i in nxt:
	            marked[i] = False
	        for i in nxt:
	            tmp_vertex_freq[trgt] += BFS(trgt,i)
	        try:
	            tmp_edge_freq[(src,trgt)] += tmp_vertex_freq[trgt]
	        except:
	            pass
	    else:
	        try:
	            tmp_edge_freq[(src,trgt)] += tmp_vertex_freq[trgt]
	        except:
	            pass

	    return tmp_vertex_freq[trgt]

	#Performing BFS from each V        
	for i in vertices:
	    marked = { int(x):True for x in vertices }
	    #Local collection of V & E freq in shortest paths 
	    tmp_vertex_freq = { v:0 for v in vertices}
	    tmp_edge_freq   = { e:0 for e in edges}

	    BFS(None,i)

	    #Updating values to GLOBAL freq counters    
	    for v in vertices:
	        vertex_freq[v] += tmp_vertex_freq[v]
	    for e in edges:
	        edge_freq[e] += tmp_edge_freq[e]

	while True:
	    for i in edge_freq:
	        if i[::-1] in edge_freq:
	            edge_freq[i] += edge_freq[i[::-1]]
	            del edge_freq[i[::-1]]
	            break
	    else:
	        break
	return vertex_freq,edge_freq



# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


def plot3D(acc,steps,neighbours):
    fig = plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')

    size = acc.shape[0]*acc.shape[1]

    xs = neighbours*10
    ys = np.array([[i]*len(neighbours) for i in range(steps)]).reshape(size,1)

    ax.scatter( xs, ys, acc.reshape(size,1), c='r', marker='^',depthshade=False)

    ax.set_xlabel('NEIGHBOURS')
    ax.set_ylabel('STEPS')
    ax.set_zlabel('ERROR')
    ax.set_title('Accuracy Curve')
    plt.savefig('output_plots/problem_2_task_6.png')
    plt.show()
    plt.clf()
    plt.close()

'''
Below are Error obtained on various M & K of Euclidiean & Cosine Matrices
ED = np.array(
[[0.25425  ,0.264375 ,0.269625 ,0.2725  ],
 [0.380125 ,0.392    ,0.4065   ,0.415375],
 [0.541875 ,0.557    ,0.568625 ,0.569   ],
 [0.801    ,0.807125 ,0.8125   ,0.810375],
 [0.88     ,0.883625 ,0.8875   ,0.881875],
 [0.8925   ,0.890125 ,0.890125 ,0.884   ],
 [0.8895   ,0.886125 ,0.887375 ,0.87775 ],
 [0.8855   ,0.882375 ,0.882125 ,0.8715  ],
 [0.88225  ,0.879625 ,0.8785   ,0.86875 ],
 [0.88175  ,0.87825  ,0.87675  ,0.868125]]
 )


CD = np.array(
[[0.108875 ,0.108875 ,0.108875 ,0.108875],
 [0.27725  ,0.30525  ,0.321125 ,0.333   ],
 [0.5195   ,0.554    ,0.561125 ,0.571625],
 [0.80875  ,0.81625  ,0.81875  ,0.817625],
 [0.8875   ,0.884375 ,0.887    ,0.8825  ],
 [0.9025   ,0.89975  ,0.900875 ,0.895875],
 [0.898375 ,0.895125 ,0.897125 ,0.89225 ],
 [0.895375 ,0.89225  ,0.895625 ,0.890625],
 [0.89375  ,0.891625 ,0.895625 ,0.891875],
 [0.893875 ,0.891875 ,0.894875 ,0.890875]]
 )
'''