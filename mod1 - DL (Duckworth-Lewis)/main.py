import os, sys, shutil, itertools as it
from datetime import datetime

import numpy as np, pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from matplotlib.ticker import StrMethodFormatter


import tensorflow
import keras
from keras.layers import Input, Dense, Multiply, Lambda
from keras.layers import Dropout, Activation, BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

e, floatx = K.epsilon(), K.floatx()

def better_precision():
	global e, floatx	
	e, floatx = e*(10**(-2)), 'float64'
	K.set_epsilon(e)
	K.set_floatx(floatx)

def exp_activation(x):
	return K.exp(x)

def sum_squared_error(y_true, y_pred):
	return K.sum(K.square(y_true - y_pred))

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
# 500 point approach, similar to all point approach
dict_mat_data = { (x,y):[mat_data[x][y]] for x in range(mat_data.shape[0]) for y in range(mat_data.shape[1])}


def write_proj_tsv():
	proj_d_df  = pd.DataFrame(data=projection_data,    columns=['OL','WIH','RM'])
	proj_md_df = pd.DataFrame(data=projection_metadata,columns=['WIH'])
	proj_d_df.to_csv( 'projection_data.tsv', index=False, sep='\t')
	proj_md_df.to_csv('projection_metadata.tsv', index=False, sep='\t')

def proj_tsv(frctn = 0.2,size=1,mode='2d'):
	'Data Points based on WIH'
	proj_points = {}
	if mode=='3d':
		class_label = list(sorted(set(y for (x,y) in dict_data)))
		for ol,wih,rm in projection_data:
			if wih in proj_points:
				proj_points[wih].append([ol,wih,rm])
			else:
				proj_points[wih] = [[ol,wih,rm]]

	elif mode=='2d':
		class_label = list(sorted(set(y for (x,y) in dict_mat_data)))
		for wih in range(mat_data.shape[1]):
			for ol in range(mat_data.shape[0]):
				if wih in proj_points:
					proj_points[wih].append([ol,wih,mat_data[ol][wih]])
				else:
					proj_points[wih] = [[ol,wih,mat_data[ol][wih]]]
	
	for i in proj_points.keys():
		for j in range(len(proj_points[i])):
			if proj_points[i][j][2]==0.0:
				proj_points[i][j][2] = np.nan

	if mode=='3d':
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for wih in sorted(proj_points):
			temp = proj_points[wih]
			np.random.shuffle(temp)
			# Taking 10% of random shuffled points to visualize
			temp = temp[:int(np.round(frctn*len(temp)))]
			xs,ys,zs = tuple(np.array(temp).T)
			ax.scatter(xs, ys, zs, label=r'$Z_{{{}}}$'.format(int(wih+1)), marker='o', s=size)
		ax.set_xlabel('Overs Left')
		plt.ylabel('Runs Scored')
		plt.yticks(np.linspace(1,10,10),np.linspace(1,10,10,dtype=int))
		ax.set_zlabel('Runs Made')
		plt.legend(bbox_to_anchor=(0.5,1), loc="lower center", ncol=len(proj_points),
		borderaxespad=0,fancybox=True,shadow=True)

	elif mode=='2d':
		for wih in proj_points:
			xs,ys,zs = tuple(np.array(proj_points[wih]).T)
			plt.plot(xs, zs,label=r'$Z_{{{}}}$'.format(int(wih+1)))
		plt.xlabel('Overs Left')
		plt.ylabel('Runs Scored')
		plt.legend(fancybox=True,shadow=True)

	title  = r'Data Points for $1^{st}$ Innings'
	title2 = 'Data Points for 1st Innings'
	plt.title(title)
	plt.grid(True)
	plt.savefig('{} {}.PNG'.format(title2,mode),dpi=400,bbox_inches='tight',format='PNG')
	plt.show()

# proj_tsv(1,10,'3d')
# proj_tsv(1,10,'2d')


# dict_mat_data = { (x,y):[mat_data[x][y]] for x in range(mat_data.shape[0]) for y in range(mat_data.shape[1])}
def NN_feed_postprocess_data(dict_form_data):
	X_wih, X_ol, Y_rm = [],[],[]
	for ol,wih in dict_form_data:
		_ = [0]*mat_data.shape[1]
		_[wih] = 1
		for i in dict_form_data[(ol,wih)]:
			X_wih.append(_)
			X_ol.append([ol])
			Y_rm.append([i])
	return [np.array(X_wih),np.array(X_ol)] , np.array(Y_rm)

def init_Z(shape):
	global Z_init
	Z_init = K.variable(mat_data.mean(axis=0)[...,np.newaxis])
	return Z_init

def init_L(shape):
	global L_init
	L_init = K.variable(np.array([1,])[...,np.newaxis])
	return L_init

def DL_model(*shapes):
	'One Hot, Followed by Real for WIH & OL respectively'
	X_inputs = list(map(Input,shapes))
	
	'Write Model Architecture using keras & Custom Activation Function'
	Z = Dense(1,input_shape=shapes[0],use_bias=False,kernel_initializer=init_Z)
	L = Dense(1,input_shape=shapes[1],use_bias=False,kernel_initializer=init_L)
	
	NUM   = L(X_inputs[1])
	DENOM = Z(X_inputs[0])
	X = Lambda( lambda x:(x[0]/x[1]) )([NUM,DENOM])
	X = Lambda( lambda x:K.exp(x) )(X)
	X = Lambda( lambda x:1-x )(X)
	X = Multiply()([ X_inputs[0] , X ])
	X = Z(X)
	model   = Model(inputs = X_inputs, outputs = X, name='DL_model')
	return model


def fit_plot(dl_model,approach = 'mean'):
	for wih in range(mat_data.shape[1]):
		_ = [0]*mat_data.shape[1]
		_[wih] = 1
		X_wih, X_ol = [], []
		for ol in range(mat_data.shape[0]):
			X_wih.append(_)
			X_ol.append([ol])
		_ = [0]*mat_data.shape[1]
		_[9] = 1
		Y_max = 1 # dl_model.predict([np.array([_,]),np.array([[50],])])
		Y_rm  = dl_model.predict([np.array(X_wih),np.array(X_ol)])
		plt.plot(Y_rm/Y_max,label=r'$Z_{{{}}}$'.format(int(wih+1)))

	title  = r'Run Scoring Fucntion of $1^{st}$ Innings'
	title2 = 'Run Scoring Fucntion of 1st Innings'
	plt.xlabel('Overs Left')
	plt.ylabel('Runs Scored')
	# plt.ylabel('Scoring Potential Left')
	plt.title(title)
	plt.grid(True)
	plt.legend(fancybox=True,shadow=True)
	plt.savefig('{} {}.PNG'.format(title2,approach),dpi=400,bbox_inches='tight',format='PNG')
	plt.show()

def dl_process(approach = 'mean',lr=0.001,factor=(1/3),epochs=10**3):
	# Data set preprocessing section
	x_train, y_train = NN_feed_postprocess_data(dict_mat_data if approach=='mean' else dict_data)

	# Model Building Section
	dl_model = DL_model((mat_data.shape[1],),(1,))

	# Training Section
	opt = keras.optimizers.Adam( lr=lr, decay=0.0 )
	dl_model.compile(optimizer=opt, loss=sum_squared_error)

	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=factor,
		patience=10, verbose=1, mode='auto', min_delta=25, cooldown=0, min_lr=0)

	history = dl_model.fit(x=x_train,y=y_train,epochs=epochs,
		batch_size=128,
		shuffle=True,callbacks=[reduce_lr])
		# batch_size=len((dict_mat_data if approach=='mean' else dict_data)),

	history = history.history

	fit_plot(dl_model)
	y_pred = dl_model.predict(x_train)
	loss = np.sum(np.square( y_train - y_pred ))
	with open('results.txt','a') as f:
		print('\nEpsilon {} and Precision {}'.format(e,floatx))
		wts = dl_model.get_weights()
		print('L scalar is\n{}'.format(wts[1][0][0]) ,file=f)
		print('Z array  is\n{}'.format(' '.join(tuple(map(str,wts[0].T[0])))) ,file=f)
		print('SSE Loss using Approach {} is {:.2f}\n'.format(approach,loss) ,file=f)
	return dl_model, history


# m1,h1 = dl_process(approach = 'mean',    lr=0.01, factor=0.99, epochs=2*10**4)
# m2,h2 = dl_process(approach = 'sep_data',lr=0.01, factor=0.99, epochs=10**3)

better_precision()

bm1,bh1 = dl_process(approach = 'mean',    lr=0.01,factor=0.99, epochs=2*10**4)
bm2,bh2 = dl_process(approach = 'sep_data',lr=0.01,factor=0.99, epochs=10**3)