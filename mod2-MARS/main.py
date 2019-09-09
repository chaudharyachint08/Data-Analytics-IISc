import os, sys, shutil, argparse, inspect
from datetime import datetime
from copy import deepcopy

import scipy, numpy as np, pandas as pd
from scipy.stats.mstats import gmean
from numpy              import mean as amean


'Reading data & converting to Matrix'

df  = pd.read_csv( os.path.join('.','..','data','01_data_mars_opposition.csv') )
mat = np.array(df)

def obj_func(row):
	res = np.zeros(mat.shape[0])
	x, y = row[:2]
	for i in range(mat.shape[0]):
		alpha = np.deg2rad( df.loc[i,'ZodiacIndex']*30 + df.loc[i,'Degree'] + 
			df.loc[i,'Minute']/60.0 + df.loc[i,'Second']/3600.0 )
		beta  = np.deg2rad( df.loc[i,'ZodiacIndexAverageSun']*30 + df.loc[i,'DegreeMean'] + 
			df.loc[i,'MinuteMean']/60.0 + df.loc[i,'SecondMean']/3600.0 )

		t1 = ( 1+np.square(np.arctan(alpha-y)) )  *  np.square(x)
		t2 = ( 2*x*(1+np.arctan(alpha-y)*np.arctan(beta-y)) )
		t3 = ( 1+np.square(np.arctan(beta-y))  )

		res[i] = np.sqrt(t1+t2+t3) / np.abs( np.arctan(alpha-y) - np.arctan(beta-y) )

	return np.log(amean(res)) - np.log(gmean(res))


bounds=((0,280),(0,4))
x_in=[8000,1000]
y0=scipy.optimize.minimize(obj_func,x_in,bounds=bounds,method='SLSQP')
print(y0)

