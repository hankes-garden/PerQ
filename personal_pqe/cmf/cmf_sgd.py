# -*- coding: utf-8 -*-
'''
Brief Description: 
    1. some implementation notes
        a. for stochastic gradient of R = PQ, if we update p_ik first, then we will use the updated p_ik in q_kj's update  

@author: jason
'''

#TODO: 1. consider how to represent sparse matrix S and R
#TODO: 2. consider how to represent unknown values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

g_dMinrmseR = 0.5
gamma0 = 0.01
power_t = 0.25
g_nMaxStep = 1000
g_dLearningRate = 0.05

def getLearningRate(gamma, nIter):
    '''
        dynamically change learning rate
    '''
    newGamma = gamma0 / pow(nIter+1, power_t)
    return newGamma

def cmf(D, S, R, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsRMSE):
    #TODO: 1. Is weight matrix a variable in loss function?
    '''
        This function factorize matrices simultaneously
        D = UP
        S = VQ
        R = bu + bv + UV
        
        params:
            D - user-profile matrix, m-by-l
            S - video-quality matrix, n-by-h (sparse, 0 for unknown)
            R - user-video matrix, m-by-n (sparse, 0 for unknown)
            arrAlphas - array of alpha, sum to 1
            arrLambdas - array of lambda for regularization
            f - number of latent factor
            nMaxStep - max iteration
            gamma - initial learning rate
            
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
    gamma = dLearningRate
    
    # D = U·V^T
    U = np.random.rand(D.shape[0], f)
    P = np.random.rand(D.shape[1], f)
    
    # S = V·Q^T
    V = np.random.rand(S.shape[0], f)
    Q = np.random.rand(S.shape[1], f)
    
    # R = B_u + B_v + U·V^T
    bu = np.random.rand(R.shape[0]) # B_u is m-by-n, with same values in a row
    bv = np.random.rand(R.shape[1]) # B_v is m-by-n, with identical values in a column
    
    # weight matrix for sparse matrices
    weightR = np.where(R<>0, 1.0, 0.0)
    weightS = np.where(S<>0, 1.0, 0.0)
    weightD = np.where(D<>0, 1.0, 0.0)
    
    nUsers = R.shape[0]
    nVideos = R.shape[1]
    for step in xrange(0, nMaxStep):
        # compute error in R
#         predR =  np.dot(U, V.T) + bu.reshape(nUsers,1) + bv.reshape((1, nVideos)) # use broadcast to add on each row/column
        predR =  np.dot(U, V.T)
        _predR = np.multiply(weightR, predR)
        errorR = np.subtract(R, _predR)
        
        # compute error in D
        predD = np.dot(U, P.T)
        _predD = np.multiply(weightD, predD)
        errorD = np.subtract(D, _predD)
        
        # compute error in S
        predS = np.dot(V, Q.T)
        _predS = np.multiply(weightS, predS)
        errorS = np.subtract(S, _predS)
        
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
        lsRMSE.append(dcRMSE)
        
        if(rmseR <= g_dMinrmseR):
            print("converged!! rmseR=%.4f" % rmseR)
            break
        
        #=======================================================================
        # debug
        print("step#: %d   r=%.4f" % (step, gamma) )
        print("    RMSE(R) = %.4f" % rmseR )
        print("    RMSE(D) = %.4f" % rmseD )
        print("    RMSE(S) = %.4f" % rmseS )
        print("------------------------------")
        #=======================================================================
        
        # change gamma
        gamma = getLearningRate(gamma, step)
        
    return bu, bv, U, P, V, Q

def visualizeRMSETrend(lsRMSE):
    df = pd.DataFrame(lsRMSE)
    df.plot(style='-o')
    plt.show()
    

def testCMF():
    # 5 users, 4 items
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
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
         [20.0, 100.8, 10.3, 0.34, 0.8, 4.0],
         [105.0, 300.8, 40.3, 0.85, 0.3, 1.0],
         [39.0, 200, 33.1, 0.45, 0.5, 6.0],
         [58.0, 60.8, 2.3, 0.13, 0.95, 8],
        ]
    S = np.array(S)
    
    # alpha for R, D, S
    arrAlphas = np.array([0.8, 0.1, 0.1])
    
    # lambda
    arrLambdas = np.array([0.4, 0.3, 0.3])
    
    # latent factors
    f = 3
    
    # CMF
    lsRMSE = []
    bu, bv, U, P, V, Q  = cmf(D, S, R, arrAlphas, arrLambdas, f, g_dLearningRate, g_nMaxStep, lsRMSE)
    
    visualizeRMSETrend(lsRMSE)
    
    print "bu= ", bu
    print "bv= ", bv
    print "U= ", U
    print "P= ", P
    print "V= ", V
    print "Q= ", Q
    
    return lsRMSE
    
        
if __name__ == '__main__':
    testCMF()