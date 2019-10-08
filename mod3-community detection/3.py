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


df = pd.read_csv(os.path.join('..','data','11_twoCirclesData.csv'))
mat = np.array(df)

def two_circle(sigma,method=1):
    def f(x,y):
        return np.exp(-np.linalg.norm(x-y)**2/sigma)
    adj_matrix = [[(f(mat[i],mat[j]) if i!=j else 0)for i in range(mat.shape[0])] for j in range(mat.shape[0])]
    
    laplacian = np.diag(np.sum(adj_matrix,axis=0))-adj_matrix
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    
    if method==1:
        reduced = eigvecs.T[:2]
        X = reduced.T
    else:
        # This one should be is Wrong
        reduced = eigvecs[:2]
        X = reduced.T
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2).fit(X)
    kmeans.labels_
    
    for lbl in set(kmeans.labels_):
        x = [mat.T[0][i] for i in range(mat.shape[0]) if lbl==kmeans.labels_[i]]
        y = [mat.T[1][i] for i in range(mat.shape[0]) if lbl==kmeans.labels_[i]]
        plt.scatter(x,y)
    plt.savefig('output_plots/problem_3.png')
    plt.show()
    plt.clf()

a,r,i = 0.001, 1, 0
two_circle(a*r**i,1)