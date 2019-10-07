import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

streams = []

def my_print(*args,**kwargs):
    for st in streams:
        print(*args,**kwargs,file=st)
 
def proj(u, v):
    # notice: this algrithm assume denominator isn't zero
    return u * np.dot(v,u) / np.dot(u,u)  
 
def GS(A):
    A = np.array(A,dtype='float64')
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i] -= proj(A[j], A[i])
    A = A.T / (A**2).sum(axis=1) **0.5
    return A.T

def euc(X,Y):
    return np.linalg.norm(X-Y)

def cos(X,Y):
    X,Y = np.array(X),np.array(Y)
    return 1-np.dot(X,Y)/(np.linalg.norm(X)*np.linalg.norm(Y))


def eigj(M):
    M = np.array(M,dtype='float64')
    num_rows     = M.shape[0]
    epoc_bound = 100
    error   = 1.0e-15       
    pi    = np.pi    
    return_flag  = 0
    # initialize eigenvalues & eigenvectors
    eigv,eigV = np.zeros(num_rows,float), np.zeros((num_rows,num_rows),float)
    for i in range(0,num_rows):
        eigV[i,i] = 1.0

    for t in range(0,epoc_bound):
         non_trace = 0;    # compute sum of off-diagonal elements in M(i,j)
         for i in range(0,num_rows):non_trace = non_trace + np.sum(np.abs(M[i,(i+1):num_rows]))
         if (non_trace < error): # diagonal form reached
              return_flag = t
              for i in range(0,num_rows):eigv[i] = M[i,i]
              break
         else:
              limit = non_trace/(num_rows*(num_rows-1)/2.0)       # average value of off-diagonal elements
              for i in range(0,num_rows-1):       # loop over lines of matrix
                   for j in range(i+1,num_rows):  # loop over columns of matrix
                       if (np.abs(M[i][j]) > limit):      # determine (ij) such that |M(i,j)| larger than average 
                                                         # value of off-diagonal elements
                           denom = M[i][i] - M[j][j]       # denominator of Eq. (3.61)
                           if (np.abs(denom) < error): phi = pi/4         # Eq. (3.62)
                           else: phi = 0.5*np.arctan(2.0*M[i][j]/denom)  # Eq. (3.61)
                           si = np.sin(phi)
                           co = np.cos(phi)
                           for k in range(i+1,j):
                               store  = M[i,k]
                               M[i][k] = M[i][k]*co + M[k][j]*si  # Eq. (3.56) 
                               M[k][j] = M[k][j]*co - store  *si  # Eq. (3.57) 
                           for k in range(j+1,num_rows):
                               store  = M[i][k]
                               M[i][k] = M[i][k]*co + M[j][k]*si  # Eq. (3.56) 
                               M[j][k] = M[j][k]*co - store  *si  # Eq. (3.57) 
                           for k in range(0,i):
                               store  = M[k,i]
                               M[k][i] = M[k][i]*co + M[k][j]*si
                               M[k][j] = M[k][j]*co - store  *si
                           store = M[i,i]
                           M[i][i] = M[i][i]*co*co + 2.0*M[i][j]*co*si + M[j][j]*si*si  # Eq. (3.58)
                           M[j][j] = M[j][j]*co*co - 2.0*M[i][j]*co*si + store  *si*si  # Eq. (3.59)
                           M[i][j] = 0.0                                            # Eq. (3.60)
                           for k in range(0,num_rows):
                                store  = eigV[k,j]
                                eigV[k][j] = eigV[k][j]*co - eigV[k][i]*si  # Eq. (3.66)
                                eigV[k][i] = eigV[k][i]*co + store     *si  # Eq. (3.67)
         return_flag = -t # in case no convergence is reached set return_flag to a negative value "-t"
    return eigv,eigV,t

buf = {}

def PCA(eig,M,X):
    global buf
    if (M,X.shape) in buf:
        pass
    else:
        _ = []
        for i in eig:
            _.extend([i]*len(eig[i]))
        _ = sorted(_,reverse=True)[:M]
        U = []
        for i in set(_):
            U.extend(eig[i])
        U = np.array(U)
        
        "Z = X*U" #(10**4,784),(784,M)
        Z = np.dot(X,U.T)
        "X'= U.T*Z" #(784,M),(M*10**4)
        X2 = np.dot(Z,U)
        
        rec_err = (np.power(np.linalg.norm(X-X2,axis=1),2).sum()/X.shape[0])**0.5
        buf[(M,X.shape)] = Z,rec_err

    del_ls = []
    for m,xs in buf:
        if xs==X.shape and m!=M:
            del_ls.append((m,xs))
    for i in del_ls:
        del buf[i]
        
    return buf[(M,X.shape)]


def kNN(eig,train_mat,M,K,test_percentage=0.2,dis=euc):
    bound = train_mat.shape[0]-int(round(len(train_mat)*test_percentage))
    test,train = train_mat[:bound],train_mat[bound:]
    X_test, Y_test  = test.T[1:].T,test.T[0]
    X_train,Y_train = train.T[1:].T,train.T[0]

    #print(*(x.shape for x in (X_train,Y_train,X_test,Y_test)))
    X2_train = PCA(eig,M,X_train)[0]
    X2_test  = PCA(eig,M,X_test) [0]

    y_pred = []
    x_train,y_train,x_test,y_test = X2_train,Y_train,X2_test,Y_test
    
    for vec in x_test:
        tmp = sorted([(dis(vec,x_train[i]),i) for i in range(len(x_train))])[:K]
        labels_tmp = {}
        for i in tmp:
            if y_train[i[1]] in labels_tmp:
                labels_tmp[ y_train[i[1]] ] += 1
            else:
                labels_tmp[ y_train[i[1]] ]  = 1
        max_freq = max(labels_tmp.values())
        valid = set(i for i in labels_tmp if labels_tmp[i]==max_freq)
        for i in tmp:
            if y_train[ i[1] ] in valid:
                pred_label = y_train[ i[1] ]
                break
        y_pred.append( pred_label )
    pred_label = np.array(y_pred,dtype='int32')
    return (np.array(y_test,dtype='int32')==pred_label).mean(),pred_label


def kNN2(eig,train,test,M,K,dis=euc):
    X_test, Y_test  = test.T[1:].T,test.T[0]
    X_train,Y_train = train.T[1:].T,train.T[0]

    X2_train = PCA(eig,M,X_train)[0]
    X2_test  = PCA(eig,M,X_test) [0]

    y_pred = []
    x_train,y_train,x_test,y_test = X2_train,Y_train,X2_test,Y_test
    
    for vec in x_test:
        tmp = sorted([(dis(vec,x_train[i]),i) for i in range(len(x_train))])[:K]
        labels_tmp = {}
        for i in tmp:
            if y_train[i[1]] in labels_tmp:
                labels_tmp[ y_train[i[1]] ] += 1
            else:
                labels_tmp[ y_train[i[1]] ]  = 1
        max_freq = max(labels_tmp.values())
        valid = set(i for i in labels_tmp if labels_tmp[i]==max_freq)
        for i in tmp:
            if y_train[ i[1] ] in valid:
                pred_label = y_train[ i[1] ]
                break
        y_pred.append( pred_label )
    pred_label = np.array(y_pred,dtype='int32')
    return (np.array(y_test,dtype='int32')==pred_label).mean(),pred_label