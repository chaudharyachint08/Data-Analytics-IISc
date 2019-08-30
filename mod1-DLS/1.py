import os, sys, shutil, argparse, inspect
from datetime import datetime
from copy import deepcopy

import numpy as np, pandas as pd


from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('text', usetex=True)
from matplotlib.ticker import StrMethodFormatter


import tensorflow
import keras
from keras.layers import Input, Dense, Add, Subtract, Multiply, Lambda
from keras.layers import Dropout, Activation, BatchNormalization
from keras.models import Model



'Libraries API based Utilities Section Begins'

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

e, floatx = K.epsilon(), K.floatx()

def better_precision():
	global e, floatx	
	e, floatx = e**2, 'float64'
	K.set_epsilon(e)
	K.set_floatx(floatx)

def sum_squared_error(y_true, y_pred):
	return K.sum(K.square(y_true - y_pred))

get_custom_objects().update({'sum_squared_error': sum_squared_error})
get_custom_objects().update({'sse': sum_squared_error})

'Libraries API based Utilities Section Ends'


# Flag for taking data for 50 overs remaining, using Innings Total
approaches              = ['sep_data'] #,'mean','sliced_data']
init_results            = False # previous results.txt file be cleared
use_better_precision    = True
before_first_over_mode  = True  # Consider (u,v) of (50,10) form also
plot_initial_data       = False # Visualization of initialization data

call_consistent         = False # if to make data_set consistent (not used)
use_consistent          = False # Not used for now
if call_consistent:
	use_consistent      = True
plots_dir               = 'plots'



'Reading Dataset & Making it More Consistent Section Begins'

data_df = pd.read_csv( os.path.join('.','..','data','04_cricket_1999to2011.csv') )
# print(data_df.columns)

def consistent_df():
	R,C = data_df.shape
	# Pointers to be set at startting and end of Innings
	innings_count, start, end = 0, 0, 0
	while start < R:
		end = start
		while (end+1<R) and (data_df['Innings'][end+1]==data_df['Innings'][start]):
			end += 1
		
		for cur in range(start+1,end):
			prev_A, prev_B = data_df['Runs'][cur-1], data_df['Innings.Total.Runs'][cur-1]
			cur_A,  cur_B  = data_df['Runs'][cur],   data_df['Innings.Total.Runs'][cur]
			next_A, next_B = data_df['Runs'][cur+1], data_df['Innings.Total.Runs'][cur+1]

			# Assuming wrong key presses, checking what can ho wrong
			if ((prev_A+prev_B) != (cur_B)) and ((cur_A+cur_B) != (next_B)):
				# Only cur_b is wrong at its place
				if (next_B-next_A)==(prev_A+prev_B):
					cur_B = prev_A+prev_B
				else:
					dc_conv = lambda x: dict( (i,str(x).count(i)) for i in set(str(x)) )
					dc_adds = lambda x: dict(())
					dc_absd = lambda x: sum(x)

					dc_prev_A, dc_prev_B = dc_conv(prev_A), dc_conv(prev_B)
					dc_cur_A,  dc_cur_B  = dc_conv(cur_A),  dc_conv(cur_B)
					dc_next_A, dc_next_B = dc_conv(next_A), dc_conv(next_B)
				data_df.loc[cur,  'Innings.Total.Runs'] = cur_B
			prev_A, prev_B = cur_A, cur_B
		#print(start,end)
		innings_count +=1
		start = end+1

if call_consistent:
	consistent_df()
	data_df.to_csv( os.path.join('.','..','data','04_cricket_1999to2011_consistent.csv'), index=False)
if use_consistent:
	data_df = pd.read_csv( os.path.join('.','..','data','04_cricket_1999to2011_consistent.csv') )

'Reading Dataset & Making it More Consistent Section Ends'

'''
col_ix = {}
for i in ['Innings','Over','Total.Overs','Innings.Total.Runs','Total.Runs','Wickets.in.Hand']:
	col_ix[i] = list(data_df.columns).index(i)
'''

'Data Storage & Collection Section Begins'

ING = np.array(data_df['Innings'])
OL  = np.array(data_df['Total.Overs'])        - np.array(data_df['Over'])
WIH = np.array(data_df['Wickets.in.Hand'])
RM  = np.array(data_df['Innings.Total.Runs']) - np.array(data_df['Total.Runs'])

# for i in ['ING','OL','WIH','RM']:
# 	print(i,eval('{}.min()'.format(i)),eval('{}.max()'.format(i)))

if before_first_over_mode:
	mat_data,mat_count  = np.zeros((51,10)), np.zeros((51,10),dtype=int)
else:
	mat_data,mat_count  = np.zeros((50,10)), np.zeros((50,10),dtype=int)
dict_data = {}
metadata_columns = ['WIH']
projection_data,projection_metadata = [],[]

for i in range(len(ING)):
	any_condition_entered = True
	if ING[i]==1:
		# Having 1 or 0 wicket in hand is same thing
		x,y,z = OL[i], max(WIH[i],1)-1, RM[i]
		last_was_1st_inning = True
	elif before_first_over_mode and (ING[i]==2) and last_was_1st_inning:
		x, y, z = 50, (10-1), data_df['Innings.Total.Runs'][i-1]
		last_was_1st_inning = False
	else:
		any_condition_entered = False
	if any_condition_entered:
		mat_data[x][y]  += z
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

'Data Storage & Collection Section Ends'



'Writing & Projection Data Section Begins'

def write_proj_tsv():
	proj_d_df  = pd.DataFrame(data=projection_data,    columns=['OL','WIH','RM'])
	proj_md_df = pd.DataFrame(data=projection_metadata,columns=['WIH'])
	proj_d_df.to_csv( 'projection_data.tsv', index=False, sep='\t')
	proj_md_df.to_csv('projection_metadata.tsv', index=False, sep='\t')

def proj_tsv(frctn = 0.2,size=1,mode='2d',gif_plot=False):
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
		ax.set_xlabel('Overs Remaining')
		plt.ylabel('Wickets In Hand')
		plt.yticks(np.linspace(1,10,10),np.linspace(1,10,10,dtype=int))
		ax.set_zlabel('Runs Made')
		plt.legend(bbox_to_anchor=(0.5,1), loc="lower center", ncol=len(proj_points),
		borderaxespad=0,fancybox=True,shadow=True)

	elif mode=='2d':
		for wih in proj_points:
			xs,ys,zs = tuple(np.array(proj_points[wih]).T)
			plt.plot(xs, zs,label=r'$Z_{{{}}}$'.format(int(wih+1)))
		plt.xlabel('Overs Remaining')
		plt.ylabel('Runs Scored')
		plt.legend(fancybox=True,shadow=True)

	title  = r'Data Points for $1^{st}$ Innings'
	title2 = 'Data Points for 1st Innings'
	plt.title(title)
	plt.grid(True)
	plt.savefig('{} {}.PNG'.format(title2,mode),dpi=400,bbox_inches='tight',format='PNG')
	# plt.show()

	# Saving Animated GIF Image
	if gif_plot and mode=='3d':
		ax.legend().set_visible(False)
		plt.legend(bbox_to_anchor=(0.5,1), loc="lower center", ncol=len(proj_points)//2,
		borderaxespad=0,fancybox=True,shadow=True)
		start_angle = 90
		def init_func(angle=start_angle):
			print('Converting for Angle',angle)
			ax.view_init(azim=angle)
		def rtt(angle):
			angle = (angle+start_angle)%360
			print('Converting for Angle',angle)
			ax.view_init(azim=angle)
		rot_animation = animation.FuncAnimation(fig=fig, func=rtt, init_func=init_func,
			frames=np.arange(0, 180+1, 1), interval=50)
		rot_animation.save('{} {}.GIF'.format(title2,mode), dpi=300, writer='imagemagick')

	

if plot_initial_data:
	proj_tsv(1,10,'2d')
	proj_tsv(1,10,'3d')
	

'Writing & Projection Data Section Ends'



'Data Encoding & Keras Model Section Begins'

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

def init_Z(shape,dtype=floatx):
	global Z_init
	Z_init = K.variable(mat_data.max(axis=0)[...,np.newaxis])
	return Z_init

def init_L(shape,dtype=floatx):
	global L_init
	L_init = K.variable(np.array([0,])[...,np.newaxis])
	return L_init

def DL_model(*shapes):
	'One Hot, Followed by Real for WIH & OL respectively'
	X_inputs = list(map(Input,shapes))
	
	'Write Model Architecture using keras & Custom Activation Function'
	Z = Dense(1,input_shape=shapes[0],use_bias=False,kernel_initializer=init_Z)
	L = Dense(1,input_shape=shapes[1],use_bias=False,kernel_initializer=init_L)
	
	NUM   = L(X_inputs[1])
	DENOM = Z(X_inputs[0])
	X = Lambda( lambda x:(-1*x[0]/x[1]) )([NUM,DENOM])
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
		# Plot for [0,50] instead of [0,49]
		for ol in range(mat_data.shape[0]+(0 if before_first_over_mode else 1)):
			X_wih.append(_)
			X_ol.append([ol])
		_ = [0]*mat_data.shape[1]
		_[9] = 1
		Y_max = 1 # dl_model.predict([np.array([_,]),np.array([[50],])])
		Y_rm  = dl_model.predict([np.array(X_wih),np.array(X_ol)])
		plt.plot(Y_rm/Y_max,label=r'$Z_{{{}}}$'.format(int(wih+1)))

	title  = r'Run Scoring Functions of $1^{st}$ Innings'
	title2 =  'Run Scoring Functions of 1st Innings'
	plt.xlabel('Overs Remaining')
	plt.ylabel('Runs Scored')
	# plt.ylabel('Scoring Potential Remaining')
	plt.title(title)
	plt.grid(True)
	plt.legend(fancybox=True,shadow=True)
	plt.savefig(os.path.join(plots_dir,'{} {}.PNG'.format(title2,approach)),
		dpi=400,bbox_inches='tight',format='PNG')
	# plt.show()
	plt.close()

def dl_process(dl_model=None, approach = 'mean',epochs=10**3,factor=(1/3),batch_size=32,
	loss='mse',opt='Adam',lr=0.001,decay=0.0,momentum=0.9,nesterov=True,RLRoP=True):
	lcls = deepcopy(locals())
	global x_train, y_train
	# Data set preprocessing section
	x_train, y_train = NN_feed_postprocess_data(dict_mat_data if approach=='mean' else dict_data)

	# Model Building Section
	if dl_model is None:
		dl_model = DL_model((mat_data.shape[1],),(1,))

	# Training Section
	if opt=='SGD':
		opt = keras.optimizers.SGD( lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
	else:
		opt = keras.optimizers.Adam( lr=lr, decay=decay )
	dl_model.compile(optimizer=opt, loss=loss)

	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=factor,
		patience=10, verbose=1, mode='auto', min_delta=0.1, cooldown=0, min_lr=0)

	if approach=='mean':
		history = dl_model.fit(x=x_train,y=y_train,epochs=epochs,
			batch_size=batch_size, shuffle=True,
			callbacks=([reduce_lr] if RLRoP else []))
	elif approach=='sep_data':
		history = dl_model.fit(x=x_train,y=y_train,epochs=epochs,
			batch_size=batch_size, shuffle=True,
			callbacks=([reduce_lr] if RLRoP else []))
	elif approach=='sliced_data':
		ls = list(range(mat_data.shape[1]))
		for i in range(epochs):
			print('Sliced Epoch',i+1)
			np.random.shuffle(ls)
			for i in ls:
				_ = mat_count.sum(axis=0)
				start_ix, end_ix = _[:i].sum(), _[:i+1].sum()
				# print((i,start_ix,end_ix),end=' ')
				x_train2, y_train2 = [[],[]], []
				x_train2[0] = x_train[0][start_ix:end_ix]
				x_train2[1] = x_train[1][start_ix:end_ix]
				y_train2    =    y_train[start_ix:end_ix]
				history = dl_model.fit(x=x_train2,y=y_train2,epochs=epochs,
					batch_size=batch_size, shuffle=True,
					callbacks=([reduce_lr] if RLRoP else []))

	# history = history.history

	x_train_r, y_train_r = NN_feed_postprocess_data(dict_data)
	y_pred_r = dl_model.predict(x_train_r)
	loss = np.sum(np.square( y_train_r - y_pred_r ))
	with open('results.txt','a') as f:
		print( *( '{:10s} {}'.format(x,lcls[x]) for x in lcls), sep='\n',file=f )
		print('Epsilon {} and Precision {}'.format(e,floatx),file=f)
		wts = dl_model.get_weights()
		# File Printing
		print('L scalar is\n{}'.format(wts[1][0][0]) ,file=f)
		print('Z array  is\n{}'.format(' '.join(tuple(map(str,wts[0].T[0])))) ,file=f)
		print(' SSE Loss using Approach {} is {:.2f}'.format(approach,loss) ,file=f)
		print('RMSE Loss using Approach {} is {:.2f}\n'.format(
			approach,np.sqrt(loss/y_pred_r.shape[0])) ,file=f)
		# Console Printing
		print('L scalar is\n{}'.format(wts[1][0][0]))
		print('Z array  is\n{}'.format(' '.join(tuple(map(str,wts[0].T[0])))))
		print(' SSE Loss using Approach {} is {:.2f}'.format(approach,loss))
		print('RMSE Loss using Approach {} is {:.2f}\n'.format(
			approach,np.sqrt(loss/y_pred_r.shape[0])))
	fit_plot(dl_model, approach = approach)
	return dl_model

'Data Encoding & Keras Model Section Ends'



# with open('results.txt','w') as f:
# 	pass


if use_better_precision:
	better_precision()

if init_results:
	with open('results.txt','w') as f:
		print('\nTotal Data Points',mat_count.sum(),end='\n\n',file=f)
else:
	with open('results.txt','a') as f:
		print('\nTotal Data Points',mat_count.sum(),end='\n\n',file=f)


if 'mean' in approaches:
	m1 = dl_process(approach = 'mean',       epochs=2*10**3,factor=0.3,batch_size=64,
		loss='mse',opt='Adam',lr=0.01,decay=0.0,momentum=0.9,nesterov=True,RLRoP=True)

if 'sep_data' in approaches:
	m2 = dl_process(approach = 'sep_data',   epochs=1*10**2,factor=0.3,batch_size=64,
		loss='mse',opt='Adam',lr=0.01,decay=0.0,momentum=0.9,nesterov=True,RLRoP=True)

if 'sliced_data' in approaches:
	m3 = dl_process(approach = 'sliced_data',epochs=1*10**1,factor=0.99,batch_size=64,
		loss='mse',opt='Adam',lr=0.01,decay=0.0,momentum=0.9,nesterov=True,RLRoP=True)