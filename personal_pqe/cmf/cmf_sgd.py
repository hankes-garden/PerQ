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
import sklearn.preprocessing as prepro
import gc

g_dMinrmseR = 0.1
g_gamma0 = 0.01
g_power_t = 0.25
g_nMaxStep = 400
g_dLearningRate = 0.01

def getLearningRate(gamma, nIter):
    '''
        dynamically change learning rate w.r.t #iteration (using sklearn default: eta = eta0 / pow(t, g_power_t) )
    '''
    return gamma # use constant learning rate for demo
#     newGamma = g_gamma0 / pow(nIter+1, g_power_t)
#     return newGamma
#     return np.log2(nIter+1) / (nIter+1.0) # set to log(n)/n


def fit(mtR, mtD, mtS, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsTrainingRMSE, dTestRatio=0.3, bDebugInfo = True):
    '''
        This function train and test the cmf model. In particular, it will 
        factorize these 3 matrices simultaneously:
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
            lsTrainingRMSE - list for RMSE of each step
            
        return:
            bu - user bias matrix, m-by-1
            bv - video bias matrix, n-by-1
            U  - User in latent factor, m-by-f
            P  - user profile in latent factor, l-by-f
            V  - video in latent factor, n-by-f
            Q  - video quality in latent factor, h-by-f
        
        Note:
            1. this function would change the content of input matrices;
            2. 255 and Nan stand for missing value in input matrices, but in the core hard
               of CMF, we use zero to represent missing value;
    '''
    print('start to initialize factor matrices...')
    #===========================================================================
    # init
    #===========================================================================
    gamma = dLearningRate
    
    # copy data to prevent modification on original data
    # or don't copy to save memory
    R = mtR
    D = mtD
    S = mtS
    
    # D = U·P^T
    U = np.random.rand(D.shape[0], f)
    P = np.random.rand(D.shape[1], f)
    
    # S = V·Q^T
    V = np.random.rand(S.shape[0], f)
    Q = np.random.rand(S.shape[1], f)
    
    # R = B_u + B_v + U·V^T + mu
    bu = np.random.rand(R.shape[0]) # B_u is m-by-n, with same values in a row, use array here for saving memory
    bv = np.random.rand(R.shape[1]) # B_v is m-by-n, with identical values in a column, use array here for saving memory

    print('start to construct weight matrices...')
    # weight matrix for sparse matrices, need to be done before filling nan
    weightR = np.where(R==255, 0.0, 1.0)
    weightD = np.where(np.isnan(D), 0.0, 1.0)
    weightS = np.where(np.isnan(S), 0.0, 1.0)
    
    # fill nan with zeros
    R[R==255] = 0
    D[np.isnan(D)] = 0.0
    S[np.isnan(S)] = 0.0
    
    #===========================================================================
    # save some samples for test
    #===========================================================================
    print('start to sample test data set...')
    arrNonzeroRows, arrNonzeroCols = np.nonzero(R)
    ix = np.random.choice(len(arrNonzeroRows), \
                          size=int(np.floor(len(arrNonzeroRows)*dTestRatio)), replace=False)
    
    # don't use these selected elements to train
    weightR[arrNonzeroRows[ix], arrNonzeroCols[ix]] = 0.0
    
    # set weight R for testing
    weightR_test = np.zeros(weightR.shape)
    weightR_test[arrNonzeroRows[ix], arrNonzeroCols[ix]] = 1.0
    
    #===========================================================================
    # training
    #===========================================================================
    print("start to train model...")
    
    # cast R into [0.0, 1.0]
    R = (R *1.0 /100.0)
    
    # compute mu, must be calculated after masking test data
    mu = ( (R*weightR).sum()*1.0)/(weightR<>0).sum()
    
    # scaling features to [0,1]
    print ('start to scale features...')
    min_max_scaler = prepro.MinMaxScaler(copy=False)
    normD = min_max_scaler.fit_transform(D)
    normS = min_max_scaler.fit_transform(S)
    
    nUsers = R.shape[0]
    nVideos = R.shape[1]
    
    #===========================================================================
    # compute initial error
    #===========================================================================
    # compute error in R (R = mu + bu + bv + U·V^T )
    predR =  (np.dot(U, V.T) + bu.reshape(nUsers,1) + bv.reshape( (1, nVideos) ) ) + mu # use broadcast to add on each row/column
    _errorR = np.subtract(R, predR)
    errorR = np.multiply(weightR, _errorR)
    
    # compute error in normD
    predD = np.dot(U, P.T)
    _errorD = np.subtract(normD, predD)
    errorD = np.multiply(weightD, _errorD)
    
    # compute error in normS
    predS = np.dot(V, Q.T)
    _errorS = np.subtract(normS, predS)
    errorS = np.multiply(weightS, _errorS)
    
    # compute rmse
    rmseR = np.sqrt( np.power(errorR, 2.0).sum() / (weightR==1.0).sum() )
    rmseD = np.sqrt( np.power(errorD, 2.0).sum() / (weightD==1.0).sum() )
    rmseS = np.sqrt( np.power(errorS, 2.0).sum() / (weightS==1.0).sum() )
    
    # save RMSE
    dcRMSE = {}
    dcRMSE['rmseR'] = rmseR
    dcRMSE['rmseD'] = rmseD
    dcRMSE['rmseS'] = rmseS
    if (lsTrainingRMSE is not None):
        lsTrainingRMSE.append(dcRMSE)
    
    # output
    if (bDebugInfo):
        print("initial step, r=%f" % (gamma) )
        print("    RMSE(R) = %f" % rmseR )
        print("    RMSE(normD) = %f" % rmseD )
        print("    RMSE(normS) = %f" % rmseS )
        print("------------------------------")
    
    
    # iterate until converge or max steps
    for step in xrange(0, nMaxStep):
        
        # get learning rate
        gamma = getLearningRate(gamma, step)
        
        #=======================================================================
        # update
        #=======================================================================
        
        # bu
        print('start to update bu...')
        for j in xrange(errorR.shape[1]):
            bu = bu + gamma* (arrAlphas[0]*errorR[:,j] - arrLambdas[0]*bu)
         
        # bv
        print('start to update bv...')
        for i in xrange(errorR.shape[0]):
            bv = bv + gamma*(arrAlphas[0]*errorR[i,:] - arrLambdas[0]*bv)
            
        oldU = np.copy(U)
        oldP = np.copy(P)
        oldV = np.copy(V)
        oldQ = np.copy(Q)
        
        # U
        print('start to update U...')
        for j in xrange(errorR.shape[1]):
            U = oldU + gamma*( arrAlphas[0]* np.dot(errorR[:, [j] ], oldV[ [j], :])
                                       + arrAlphas[1]*np.dot(errorD, oldP)\
                                       - arrLambdas[0]*oldU )
            if (j % 1000 == 0):
                print('-->U: %.2f%%' % (j*100.0/errorR.shape[1]) )
        
        # P
        print('start to update P...')
        for i in xrange(errorD.shape[0]):
            P = oldP + gamma* ( arrAlphas[1]* np.dot(errorD[[i],:].T, oldU[[i],:]) - arrLambdas[1]*oldP)
        
        # V
        print('start to update V...')
        for i in xrange(errorR.shape[0]):
            V = oldV + gamma*( arrAlphas[0] * np.dot(errorR[[i], :].T, oldU[[i], :]) \
                                       + arrAlphas[2]* np.dot(errorS, oldQ) \
                                       - arrLambdas[0]*oldV )
            if (i % 1000 == 0):
                print('-->V: %.2f%%' % (i*100.0/errorR.shape[0]) )
        
        # Q
        print('start to update Q...')
        for i in xrange(errorS.shape[0]):
            Q = oldQ + gamma* (arrAlphas[2]* np.dot(errorS[[i],:].T, oldV[[i],:]) - arrLambdas[2]*oldQ )
        
        #=======================================================================
        # update error
        #=======================================================================
        print('start to compute error...')
        # compute error in R
        predR =  (np.dot(U, V.T) + bu.reshape(nUsers,1) + bv.reshape( (1, nVideos) ) ) + mu # use broadcast to add on each row/column
        _errorR = np.subtract(R, predR)
        errorR = np.multiply(weightR, _errorR)
        
        # compute error in normD
        predD = np.dot(U, P.T)
        _errorD = np.subtract(normD, predD)
        errorD = np.multiply(weightD, _errorD)
        
        # compute error in normS
        predS = np.dot(V, Q.T)
        _errorS = np.subtract(normS, predS)
        errorS = np.multiply(weightS, _errorS)
        
        rmseR = np.sqrt( np.power(errorR, 2.0).sum() / (weightR<>0).sum() )
        rmseD = np.sqrt( np.power(errorD, 2.0).sum() / (weightD<>0).sum() )
        rmseS = np.sqrt( np.power(errorS, 2.0).sum() / (weightS<>0).sum() )
        
        # save RMSE
        dcRMSE = {}
        dcRMSE['rmseR'] = rmseR
        dcRMSE['rmseD'] = rmseD
        dcRMSE['rmseS'] = rmseS
        if (lsTrainingRMSE is not None):
            lsTrainingRMSE.append(dcRMSE)
            
        #=======================================================================
        if (bDebugInfo):
            print("step#: %d   r=%f" % (step, gamma) )
            print("    RMSE(R) = %f" % rmseR )
            print("    RMSE(normD) = %f" % rmseD )
            print("    RMSE(normS) = %f" % rmseS )
            print("------------------------------")
        #=======================================================================
        
        
        #=======================================================================
        # check convergence
        #=======================================================================
        if(rmseR <= g_dMinrmseR):
            print("converged!! rmseR=%.4f" % rmseR)
            break
        
        #=======================================================================
        # time to clean memory
        #=======================================================================
        print('time to release memory...')
        gc.collect()
        
    #===========================================================================
    # test
    #===========================================================================
    print("start to test...")
    predR_test =  (np.dot(U, V.T) + bu.reshape(nUsers,1) + bv.reshape( (1, nVideos) ) ) + mu # use broadcast to add on each row/column
    _errorR_test = np.subtract(R, predR_test)
    errorR_test = np.multiply(weightR_test, _errorR_test)
    rmseR_test = np.sqrt( np.power(errorR_test, 2.0).sum() / (weightR_test<>0).sum() )
    
    print("-->traning: %f, test: %f" % (rmseR_test, lsTrainingRMSE[-1]['rmseR']) )
    
    
    return mu, bu, bv, U, P, V, Q, rmseR_test

def visualizeRMSETrend(lsRMSE):
    '''
        This function visualize the changing of RMSE
    '''
    import matplotlib.pyplot as plt
    df = pd.DataFrame(lsRMSE)
    df = df[['rmseD', 'rmseS', 'rmseR']]
    df.columns = ['D', 'S', 'R']
    lsSamples = range(0,1000,10)
    lsSamples.append(999)
    ax = df.plot(style=['--','-.', '-' ], ylim=(0.0, 1.0))
    ax.set_ylabel('RMSE')
    plt.show()
    

