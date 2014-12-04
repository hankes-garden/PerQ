# -*- coding: utf-8 -*-
'''
Brief Description: 
    This model implement Collective Matrix factorization (http://dl.acm.org/citation.cfm?id=1401969).
    Insteading using stochastic-optimized newton's method solution, we solve the minimization of the
    loss function by a stochastic gradient descent

@author: jason
'''

#TODO: 1. consider how to represent sparse matrix S and R
#TODO: 2. consider how to represent unknown values
#TODO: 1. Is weight matrix a variable in loss function?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prepro
from matplotlib.pyplot import yticks

g_dMinrmseR = 0.05
g_gamma0 = 0.01
g_power_t = 0.25
g_nMaxStep = 1000
g_dLearningRate = 0.01

def getLearningRate(gamma, nIter):
    '''
        dynamically change learning rate w.r.t #iteration (using sklearn default: eta = eta0 / pow(t, g_power_t) )
    '''
    return 0.01
#     newGamma = g_gamma0 / pow(nIter+1, g_power_t)
#     return newGamma
#     return np.log2(nIter+1) / (nIter+1.0) # set to log(n)/n


def fit(mtR, mtD, mtS, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsRMSE=None):
    '''
        This function factorize matrices simultaneously
        D = U·P^T
        S = V·Q^T
        R = bu + bv + U·V^T
        
        params:
            R - user-video matrix, m-by-n (sparse, 0 for unknown)
            D - user-profile matrix, m-by-l (dense)
            S - video-quality matrix, n-by-h (dense)
            arrAlphas - re-construction weights for R, D, S
            arrLambdas - lambda for regularization for R, D, S factorization
            f - number of latent factor
            dLearningRate - initial learning rate
            nMaxStep - max iteration
            lsRMSE - list for RMSE of each step
            
        return
            bu - user bias matrix, m-by-1
            bv - video bias matrix, n-by-1
            U  - User in latent factor, m-by-f
            P  - user profile in latent factor, l-by-f
            V  - video in latent factor, n-by-f
            Q  - video quality in latent factor, h-by-f
    '''
    #===========================================================================
    # init
    #===========================================================================
    R = np.copy(mtR)
    D = np.copy(mtD)
    S = np.copy(mtS)
    
    gamma = dLearningRate
    
    # D = U·V^T
    U = np.random.rand(D.shape[0], f)
    P = np.random.rand(D.shape[1], f)
    
    # S = V·Q^T
    V = np.random.rand(S.shape[0], f)
    Q = np.random.rand(S.shape[1], f)
    
    # R = B_u + B_v + U·V^T
    bu = np.random.rand(R.shape[0]) # B_u is m-by-n, with same values in a row, use array here for saving memory
    bv = np.random.rand(R.shape[1]) # B_v is m-by-n, with identical values in a column, use array here for saving memory
    
    # weight matrix for sparse matrices
    weightR = np.where(np.isnan(R), 0.0, 1.0)
    weightS = np.where(np.isnan(S), 0.0, 1.0)
    weightD = np.where(np.isnan(D), 0.0, 1.0)
    
    R[np.isnan(R)] = 0.0
    D[np.isnan(D)] = 0.0
    S[np.isnan(S)] = 0.0
    
    
    #===========================================================================
    # normalize (R does not need to normalize)
    #===========================================================================
    normD = prepro.normalize(D, axis=1)
    normS = prepro.normalize(S, axis=1)
    
    nUsers = R.shape[0]
    nVideos = R.shape[1]
    for step in xrange(0, nMaxStep):
        
        # get learning rate
        gamma = getLearningRate(gamma, step)
        
        # compute error in R
        predR =  np.dot(U, V.T) + bu.reshape(nUsers,1) + bv.reshape((1, nVideos)) # use broadcast to add on each row/column
#         predR =  np.dot(U, V.T)
        _predR = np.multiply(weightR, predR)
        errorR = np.subtract(R, _predR)
        
        # compute error in normD
        predD = np.dot(U, P.T)
        _predD = np.multiply(weightD, predD)
        errorD = np.subtract(normD, _predD)
        
        # compute error in normS
        predS = np.dot(V, Q.T)
        _predS = np.multiply(weightS, predS)
        errorS = np.subtract(normS, _predS)
        
        #=======================================================================
        # update
        #=======================================================================
        # bu
        for j in xrange(errorR.shape[1]):
            bu = bu + gamma*arrAlphas[0]* (errorR[:,j] - arrLambdas[0]*bu)
         
        # bv
        for i in xrange(errorR.shape[0]):
            bv = bv + gamma*arrAlphas[1]*(errorR[i,:] - arrLambdas[1]*bv)
            
        oldU = np.copy(U)
        oldP = np.copy(P)
        oldV = np.copy(V)
        oldQ = np.copy(Q)
        
        # U
        for i in xrange(errorR.shape[0]):
            U[i,:] = oldU[i,:] + gamma*( \
                                      arrAlphas[0]*( np.dot(errorR[[i],:], oldV)-arrLambdas[0]*oldU[[i],:] ) \
                                      + arrAlphas[1]*np.dot(errorD[[i],:], oldP) \
                                    )
        
        # P
        for i in xrange(errorD.shape[0]):
            P = oldP + gamma*arrAlphas[1]*(np.dot(errorD[[i],:].T, oldU[[i],:])-arrLambdas[1]*oldP)
        
        # V
        for i in xrange(errorR.shape[1]):
            V[i,:] = oldV[i,:] + gamma*( \
                                     arrAlphas[0]*( np.dot(errorR[:,[i]].T, oldU)-arrLambdas[0]*oldV[[i],:] )
                                     + arrAlphas[2]*np.dot(errorS[[i],:], oldQ) \
                                    )
        
        # Q
        for i in xrange(errorS.shape[0]):
            Q = oldQ + gamma*arrAlphas[2]*(np.dot(errorS[[i],:].T, oldV[[i],:])-arrLambdas[2]*oldQ)
        
        #=======================================================================
        # check convergence
        #=======================================================================
        rmseR = np.sqrt( np.power(errorR, 2.0).sum() / (weightR<>0).sum() )
        rmseD = np.sqrt( np.power(errorD, 2.0).sum() / (weightD<>0).sum() )
        rmseS = np.sqrt( np.power(errorS, 2.0).sum() / (weightS<>0).sum() )
        # save RMSE
        dcRMSE = {}
        dcRMSE['rmseR'] = rmseR
        dcRMSE['rmseD'] = rmseD
        dcRMSE['rmseS'] = rmseS
        if (lsRMSE is not None):
            lsRMSE.append(dcRMSE)
        
        if(rmseR <= g_dMinrmseR):
            print("converged!! rmseR=%.4f" % rmseR)
            break
        
        #=======================================================================
        # debug
        print("step#: %d   r=%.4f" % (step, gamma) )
        print("    RMSE(R) = %.4f" % rmseR )
        print("    RMSE(normD) = %.4f" % rmseD )
        print("    RMSE(normS) = %.4f" % rmseS )
        print("------------------------------")
        #=======================================================================
        
    return bu, bv, U, P, V, Q

def visualizeRMSETrend(lsRMSE):
    '''
        This function visualize the changing of RMSE
    '''
    df = pd.DataFrame(lsRMSE)
    df = df[['rmseD', 'rmseS', 'rmseR']]
    df.columns = ['D', 'S', 'R']
    lsSamples = range(0,1000,10)
    lsSamples.append(999)
    ax = df.iloc[lsSamples].plot(yticks=np.arange(0, 1, 0.05), ylim=(0.0,1.0), style=['--','-.', '-' ])
    ax.set_ylabel('RMSE')
    plt.show()
    

def testCMF():
    # 5 users, 4 items
    R = [
         [0.2,0.3,0.0,1.0],
         [0.9,0.0,0.0,0.1],
         [0.1,0.1,0.0,0.9],
         [1.0,0.0,0.0,0.4],
         [0.0,0.1,0.6,0.3],
        ]
    R = np.array(R)

    # 5 users, 3 profiles
    D = [
         [16,10,3.5],
         [40,18,1.0],
         [22,50,2.8],
         [28,80,2.5],
         [36,80,1.8],
        ]
    D = np.array(D)
    
    # 4 items, 6 features
    S = [
         [20.0, 20.8, 10.3, 0.34, 0.8, 40.0],
         [105.0, 40.8, 40.3, 0.85, 0.3, 10.0],
         [39.0, 80.0, 33.1, 0.45, 0.5, 60.0],
         [58.0, 30.8, 2.3, 0.13, 0.95, 80.0],
        ]
    S = np.array(S)
    
    # alpha for R, D, S
    arrAlphas = np.array([0.8, 0.1, 0.1])
    
    # lambda
    arrLambdas = np.array([0.4, 0.3, 0.3])
    
    # latent factors
    f = 3
    
    # normalize
    normD = prepro.normalize(D, axis=1)
    normS = prepro.normalize(S, axis=1)
    
    # CMF
    lsRMSE = []
    bu, bv, U, P, V, Q  = fit(R, normD, normS, arrAlphas, arrLambdas, f, g_dLearningRate, 1000, lsRMSE)
    
    print "bu= ", bu
    print "bv= ", bv
    print "U= ", U
    print "P= ", P
    print "V= ", V
    print "Q= ", Q
    
    visualizeRMSETrend(lsRMSE)
    
    return lsRMSE
    
        
if __name__ == '__main__':
    testCMF()