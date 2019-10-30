#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, shutil, argparse, inspect
from datetime import datetime
from copy import deepcopy

import numpy as np, pandas as pd
from scipy.stats import f as fisher_f
from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('text', usetex=False)
from matplotlib.ticker import StrMethodFormatter

import warnings
warnings.filterwarnings('ignore')


# In[2]:


PPL, CAT = 12,4
data = pd.read_csv(os.path.join('.','..','data','Raw Data_GeneSpring.txt'),sep='\t')


# In[3]:


D1, D2 = np.zeros((PPL*CAT,CAT)), np.zeros((PPL*CAT,CAT))


# In[4]:


D1[0*PPL:1*PPL], D2[0*PPL:1*PPL] = np.array([1.0,0.0,0.0,0.0],dtype='float64'), np.array([1.0,0.0,1.0,0.0],dtype='float64')
D1[1*PPL:2*PPL], D2[1*PPL:2*PPL] = np.array([0.0,1.0,0.0,0.0],dtype='float64'), np.array([1.0,0.0,0.0,1.0],dtype='float64')
D1[2*PPL:3*PPL], D2[2*PPL:3*PPL] = np.array([0.0,0.0,1.0,0.0],dtype='float64'), np.array([0.0,1.0,1.0,0.0],dtype='float64')
D1[3*PPL:4*PPL], D2[3*PPL:4*PPL] = np.array([0.0,0.0,0.0,1.0],dtype='float64'), np.array([0.0,1.0,0.0,1.0],dtype='float64')


# In[5]:


M1 = np.linalg.multi_dot([D1,np.linalg.pinv(np.dot(D1.T,D1)),D1.T])
M2 = np.linalg.multi_dot([D2,np.linalg.pinv(np.dot(D2.T,D2)),D2.T])
RANK1, RANK2 = tuple(map(np.linalg.matrix_rank,(M1,M2)))
NUM, DENOM = (M1-M2), (np.diag(np.ones(M1.shape[0]))-M1)
diff1, diff2  = RANK1 - RANK2, PPL*CAT - RANK1


# In[6]:


# Task - 1 : Generating p-values
def my_map(prm):
    LIST   = prm[1:1+PPL*CAT]
    NUM2, DENOM2   = np.linalg.multi_dot([LIST.T,NUM,  LIST]), np.linalg.multi_dot([LIST.T,DENOM,LIST])
    val = (NUM2*(PPL*CAT - RANK1))/(DENOM2*(RANK1-RANK2))
    if val:
        return val
    else:
        return 0

data['f_val'] = data.apply( my_map , axis=1 )
data['p_val'] = 1 - fisher_f(diff1,diff2,0).cdf( data['f_val'] )

p_vals = np.array(sorted(data['p_val']))
p_vals = p_vals[~np.isnan(p_vals)]


# In[7]:


# Task-2 : Generating Histogram of p-values
data['p_val'].hist()
plt.show()


# In[9]:


# Task-4 : Using FDR cut-off of 0.05 for shortlisting rows, and 
FDR_cutoff = 0.05
shrt_ls_rows = data['p_val']<FDR_cutoff


# In[10]:


# Task-5 :Creating list of GeneSymbols
gene_sym = data[shrt_ls_rows]['GeneSymbol']
gene_sym = gene_sym.replace(np.nan,'Nan',regex=True)
print(list(gene_sym))


# In[11]:


# Task-6 : Intersecting with various gene lists
files = ('XenobioticMetabolism1.txt','FreeRadicalResponse.txt','DNARepair1.txt','NKCellCytotoxicity.txt')
ext_data,res_sym = [], []
for f_name in files:
    ext_data.append( pd.read_csv(os.path.join('.','..','data',f_name ),sep='\t') )
    res_sym.append( list(pd.Series(list(set(gene_sym).intersection(set(list(ext_data[-1][ext_data[-1].columns[0]])))))) )


# In[12]:


for name, count, genes in zip(files,map(len,res_sym),res_sym):
    print('File Name is',name )
    print('Intersection Count is',count)
    print('Genes from Intersection are\n',genes,end='\n\n')


# In[13]:


# Task-7 : Finding the difference in response of gene in each gender, differentiating smoker or not
def comp(false_ls, true_ls,gender):
    points = 100
    # Fit a normal distribution to the data:
    f_mu, f_std = np.mean(false_ls), np.std(false_ls) 
    t_mu, t_std = np.mean(true_ls ), np.std(true_ls )

    # Plot the PDF.
    xmin, xmax = min(false_ls+true_ls), max(false_ls+true_ls)
    x = np.linspace(xmin, xmax, points)

    f_p = norm.pdf(x, f_mu, f_std)
    t_p = norm.pdf(x, t_mu, t_std)

    plt.plot(x, f_p, 'g', linewidth=2)
    plt.plot(x, t_p, 'r', linewidth=2)

    plt.title('{} - {}'.format(box_title,gender))
    plt.savefig(os.path.join('plots','{} - {}.PNG'.format(box_title,gender)),dpi=400,bbox_inches='tight',format='PNG')
#     plt.show()
    plt.close()

#     f_1, f_3 = np.quantile(f_p,0.16), np.quantile(f_p,0.84)
#     t_1, t_3 = np.quantile(t_p,0.16), np.quantile(t_p,0.84)
    
#     f_p_s = [x for x in f_p if ( f_1<=x and x<=f_3 )]
#     t_p_s = [x for x in t_p if ( t_1<=x and x<=t_3 )]
#     if np.mean(f_p_s) < np.mean(t_p_s):
#         print('Lesser Response in Smokers')
#     else:
#         print('More   Response in Smokers')



if 'plots' not in os.listdir():
    os.mkdir('plots')

tmp_ls = list(data['GeneSymbol'])
for ix1,sym_ls in enumerate(res_sym):
    print('\n\nFile Name : ',files[ix1])
    for sym in sym_ls:
        print('\n',sym)
        ix2 = tmp_ls.index(sym)
        dct = {}
        typ_ls = ('MN','MS','FN','FS')
        for ix3,typ in enumerate(typ_ls):
            dct[typ] = list(data.iloc[ix2,1+ix3*PPL:1+(ix3+1)*PPL])
        fig, ax = plt.subplots()
        plot_data = [dct[k] for k in typ_ls] 
                
        # Box plot of Data
        ax.boxplot(plot_data)#, showfliers=False)
        plt.xticks(np.arange(1,len(typ_ls)+1),typ_ls)
        
        box_title = '_'.join((files[ix1],sym))
        plt.title(box_title)
        plt.savefig(os.path.join('plots','{}.PNG'.format(box_title)),dpi=400,bbox_inches='tight',format='PNG')
#         plt.show()
        plt.close()

        for gender in ('M','F'):
            dct2 = {}
            for k in dct.keys():
                if gender in k:
                    dct2[k] = dct[k]
            false_ls = dct2[gender+'N']
            true_ls  = dct2[gender+'S']
            false_md = np.median(false_ls)
            true_md  = np.median(true_ls )
                
            print('Gender',gender,end='\t')
            if false_md < true_md:
                print('More Response in Non-Smokers')
            else:
                print('More Response in Smokers')
                
            # Gaussian KDE approximation of Data
#             sns.distplot(false_ls,color='g',bins=PPL)
#             sns.distplot(true_ls, color='r',bins=PPL)
#             plt.show()

            # Gaussian Approximation of data & Line visualization
            comp(false_ls,true_ls,gender)
plt.close()


# In[ ]:




