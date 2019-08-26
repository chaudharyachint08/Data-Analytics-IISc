import os, sys, shutil, itertools as it
from datetime import datetime

import numpy as np, pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


import tensorflow
import keras
from keras.layers import Input, Add, Subtract, Average, Multiply, Concatenate
from keras.layers import Dropout, Activation, BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def exp_activation(x):
	return K.exp(x)
get_custom_objects().update({'exp_activation': Activation(exp_activation)})


data_df = pd.read_csv('04_cricket_1999to2011.csv')
print(data_df.columns)

col_ix = {}
for i in ['Innings','Over','Total.Overs','Innings.Total.Runs','Total.Runs','Wickets.in.Hand']:
	col_ix[i] = list(data_df.columns).index(i)

ING = np.array(data_df['Innings'])
OL  = np.array(data_df['Total.Overs'])        - np.array(data_df['Over'])
WIH = np.array(data_df['Wickets.in.Hand'])
RM  = np.array(data_df['Innings.Total.Runs']) - np.array(data_df['Total.Runs'])

for i in ['ING','OL','WIH','RM']:
	print(i,eval('{}.min()'.format(i)),eval('{}.max()'.format(i)))

'''
ING 1 2 
OL 0 49 # can be used as 50 columns
WIH 0 10
RM -1 435
'''

mat_data,mat_count  = np.zeros((50,10)), np.zeros((50,10))
dict_data = {}
metadata_columns = ['WIH']
projection_data,projection_metadata = [],[]

for i in range(len(ING)):
	if ING[i]==1:
		# Having 1 or 0 wicket in hand is same thing
		x,y = OL[i], max(WIH[i],1)-1
		mat_data[x][y]  += RM[i]
		mat_count[x][y] += 1
		if (x,y) in dict_data:
			dict_data[(x,y)].append(RM[i])
		else:
			dict_data[(x,y)] = [ RM[i], ]
		projection_data.append( (x,y,RM[i]) )
		projection_metadata.append( y )
mat_data = np.where(mat_count,mat_data/mat_count,0)

def write_proj_tsv():
	proj_d_df  = pd.DataFrame(data=projection_data,    columns=['OL','WIH','RM'])
	proj_md_df = pd.DataFrame(data=projection_metadata,columns=['WIH'])
	proj_d_df.to_csv( 'projection_data.tsv', index=False, sep='\t')
	proj_md_df.to_csv('projection_metadata.tsv', index=False, sep='\t')

def proj_tsv(frctn = 0.2,size=1):
	'3D prokection of Data Points based on WIH'
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	class_label = list(sorted(set(y for (x,y) in dict_data)))
	proj_points = {}
	for ol,wih,rm in projection_data:
		if wih in proj_points:
			proj_points[wih].append((ol,wih,rm))
		else:
			proj_points[wih] = [(ol,wih,rm)]

	for wih in proj_points:
		temp = proj_points[wih]
		np.random.shuffle(temp)
		# Taking 10% of random shuffled points to visualize
		temp = temp[:int(np.round(frctn*len(temp)))]
		xs,ys,zs = tuple(np.array(temp).T)
		ax.scatter(xs, ys, zs, label=str(wih), marker='o', s=size)
	ax.set_xlabel('Overs Left')
	ax.set_ylabel('Wickets In Hand')
	plt.yticks(np.linspace(1,10,10),np.linspace(1,10,10,dtype=int))
	ax.set_zlabel('Runs Made')
	plt.legend(bbox_to_anchor=(0.5,1), loc="lower center", ncol=len(proj_points),
		borderaxespad=0,fancybox=True,shadow=True)
	plt.title('Data Points for 1st Innings')
	
	plt.grid(True)

	plt.show()

# proj_tsv(1,10)


# 500 point approach, similar to all point approach
dict_mat_data = { (x,y):[mat_data[x][y]] for x in range(mat.shape[0]) for y in range(mat.shape[1])}


def DL_model(*shapes):
	X_inputs = list(map(Input,shapes))
	
	'Write Model Architecture using keras & Custtom Activation Function'


	model   = Model(inputs = X_inputs, outputs = X, name='DL_model')
	return model