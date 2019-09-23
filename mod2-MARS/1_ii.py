import os, sys, shutil, argparse, inspect
from datetime import datetime
from copy import deepcopy

import scipy, numpy as np, pandas as pd
from scipy.stats.mstats import gmean
from numpy              import mean as amean
import matplotlib.pyplot as plt


'Reading data & converting to Matrix'

df  = pd.read_csv( os.path.join('.','..','data','01_data_mars_opposition.csv') )

def get_global_alpha_beta(df):
    res_alpha, res_beta = [],[]
    for i in range(df.shape[0]):
        alpha = np.deg2rad( df.loc[i,'ZodiacIndex']*30 + df.loc[i,'Degree'] + 
            df.loc[i,'Minute']/60.0 + df.loc[i,'Second']/3600.0 )
        beta  = np.deg2rad( df.loc[i,'ZodiacIndexAverageSun']*30 + df.loc[i,'DegreeMean'] + 
            df.loc[i,'MinuteMean']/60.0 + df.loc[i,'SecondMean']/3600.0 )
        res_alpha.append( alpha )
        res_beta.append(  beta  )
    return np.array(res_alpha), np.array(res_beta)

def indx_rad_phi(point,ix):
    x, y = point
    p1, p2, b = np.sin(beta[ix]-y), np.sin(alpha[ix]-y), np.cos(beta[ix]-alpha[ix])
    n1, n2, n3, d = 2*x*p1*p2*b , np.square(p1) , np.square(x*p2) , (1-np.square(b))
    rad = np.sqrt( sum([n1,n2,n3]) / d )
    phi = np.arcsin((x*p2)/rad)+alpha[ix]
    return rad, phi

def obj_func(point,args):
    ls_radius = np.array( [indx_rad_phi(point,i)[0] for i in range(df.shape[0])] )
    return np.log(amean(ls_radius)) - np.log(gmean(ls_radius))

s1, s2 = 1.15,0.25
alpha, beta = get_global_alpha_beta(df)
bounds = [(0,np.inf),(-np.pi,np.pi)]

opt_res = scipy.optimize.minimize(obj_func,(s1,s2),args=[alpha,beta],bounds=bounds,method='L-BFGS-B')
print( 'Minimum Value of Function Obtained is {}'.format(opt_res['fun']) )
print( 'Points providing mimum value\nx = {}\ny = {}'.format(opt_res['x'][0], opt_res['x'][1]) )

# Plotting Points on Orbit
plt_X, plt_Y = [], []
for i in range(df.shape[0]):
    rad, phi = indx_rad_phi((opt_res['x'][0], opt_res['x'][1]),i)
    plt_X.append( rad * np.cos(phi) )
    plt_Y.append( rad * np.sin(phi) )
title = 'Orbital Positions'
plt.scatter(plt_X,plt_Y,c='r')
plt.title(title)
plt.savefig('{}.PNG'.format(title),dpi=400,format='PNG')