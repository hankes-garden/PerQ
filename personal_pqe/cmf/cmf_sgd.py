# -*- coding: utf-8 -*-
'''
Brief Description: 
    This model implement Collective Matrix factorization (http://dl.acm.org/citation.cfm?id=1401969).
    Insteading using stochastic-optimized newton's method solution, we solve the minimization of the
    loss function by a stochastic gradient descent

@author: jason
'''

import numpy as np
import pandas as pd
import sklearn.preprocessing as prepro
import gc

g_dEpsilon = 0.1
g_gamma0 = 0.01
g_power_t = 0.25


def getLearningRate(gamma, nIter):
    '''
        dynamically change learning rate w.r.t #iteration (using sklearn default: eta = eta0 / pow(t, g_power_t) )
    '''
    return gamma # use constant learning rate for demo

#     newGamma = g_gamma0 / pow(nIter+1, g_power_t)
#     return newGamma

#     return np.log2(nIter+1) / (nIter+1.0) # set to log(n)/n

def computeResidualError(R, D, S, U, V, P, Q, mu, weightR, weightD, weightS):
    #===========================================================================
    # compute initial error
    #===========================================================================
    # compute error in R (R = mu + bu + bv + U·V^T )
    # predR =  (np.dot(U, V.T) + bu + bv ) + mu # use broadcast to add on each row/column
    predR =  np.dot(U, V.T) + mu
    _errorR = np.subtract(R, predR)
    errorR = np.multiply(weightR, _errorR)
    
    # compute error in normD
    predD = np.dot(U, P.T)
    _errorD = np.subtract(D, predD)
    errorD = np.multiply(weightD, _errorD)
    
    # compute error in normS
    predS = np.dot(V, Q.T)
    _errorS = np.subtract(S, predS)
    errorS = np.multiply(weightS, _errorS)
    
    # compute rmse
    rmseR = np.sqrt( np.power(errorR, 2.0).sum() / weightR.sum() )
    rmseD = np.sqrt( np.power(errorD, 2.0).sum() / weightD.sum() )
    rmseS = np.sqrt( np.power(errorS, 2.0).sum() / weightS.sum() )
    
    dTotalLost = (arrAlphas[0]/2.0) * np.power(np.linalg.norm(errorR, ord='fro'), 2.0) \
            + (arrAlphas[1]/2.0) * np.power(np.linalg.norm(errorD, ord='fro'), 2.0) \
            + (arrAlphas[2]/2.0) * np.power(np.linalg.norm(errorS, ord='fro'), 2.0) \
            + (arrLambdas[0]/2.0) * ( np.power(np.linalg.norm(U, ord='fro'), 2.0) + np.power(np.linalg.norm(V, ord='fro'), 2.0) ) \
            + (arrLambdas[1]/2.0) * np.power(np.linalg.norm(P, ord='fro'), 2.0) \
            + (arrLambdas[2]/2.0) * np.power(np.linalg.norm(Q, ord='fro'), 2.0)
            
    return errorR, errorD, errorS, rmseR, rmseD, rmseS, dTotalLost

def computeParitialGraident(errorR, errorD, errorS, oldU, oldV, oldP, oldQ):
    # U
    gradU = ( -1.0*arrAlphas[0]*np.dot(errorR, oldV) - arrAlphas[1]*np.dot(errorD, oldP) \
                      + arrLambdas[0]*oldU )
    # P
    gradP = ( -1.0*arrAlphas[1]*np.dot(errorD.T, oldU) + arrLambdas[1]*oldP )
    
    # V
    gradV = ( -1.0*arrAlphas[0]*np.dot(errorR.T, oldU) - arrAlphas[2]*np.dot(errorS, oldQ) \
                      + arrLambdas[0]*oldV )
    # Q
    gradQ = ( -1.0 * arrAlphas[2]*np.dot(errorS.T, oldV) + arrLambdas[2]*oldQ )
    
    return gradU, gradV, gradP, gradQ

def fit_test(mtR, mtD, mtS, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsTrainingRMSE, dTestRatio=0.3, bDebugInfo = True):
    '''
        This function train and test the cmf model. In particular, it will 
        factorize these 3 matrices simultaneously:
            D = U·P^T
            S = V·Q^T
            R = mu + bu + bv + U·V^T
        
        params:
            mtR - user-video matrix, m-by-n (sparse matrix, each element is a unsigned integer,  255 for unknown, known ones belong to 0~100 )
            mtD - user-profile matrix, m-by-l (dense, np.nan for unknown)
            mtS - video-quality matrix, n-by-h (dense, np.nan for unknown)
            arrAlphas - re-construction weights for R, D, S
            arrLambdas - lambdas for regularization
            f - number of latent factors
            dLearningRate - initial learning rate
            nMaxStep - max iteration steps
            lsTrainingRMSE - list to store RMSE of each training step
            
        return:
            mu - average rating
            bu - user bias matrix, m-by-1
            bv - video bias matrix, n-by-1
            U  - User in latent factor, m-by-f
            P  - user profile in latent factor, l-by-f
            V  - video in latent factor, n-by-f
            Q  - video quality in latent factor, h-by-f
            rmseR_test - RMSE of R matrix in test
        
        Note:
            1. this function would change the content of input matrices ;
            2. input matrices may use both np.nan and 255 to represent missing values, 
               but in the computing core of CMF, we use 0 to represent missing value;
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
    

    print('start to construct weight matrices...')
    # weight matrix for sparse matrices, need to be done before filling nan
    weightR = np.where(R==255, 0.0, 1.0)
    weightD = np.where(np.isnan(D), 0.0, 1.0)
    weightS = np.where(np.isnan(S), 0.0, 1.0)
    
    # fill missing values with 0 (need to done before selecting test set)
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
    # transform input matrices
    #===========================================================================
    # cast R into [0.0, 1.0]
    R = (R *1.0 /100.0)
    
    # compute mu ( for mean normalization), must be calculated after masking test data
    mu = ( (R*weightR).sum(axis=1)*1.0) / weightR.sum(axis=1)
    mu = np.reshape(mu, (mu.shape[0], 1) ) 
    
    # scaling features to [0,1]
    print ('start to scale features...')
    min_max_scaler = prepro.MinMaxScaler(copy=False)
    normD = min_max_scaler.fit_transform(D)
    normS = min_max_scaler.fit_transform(S)
#     normD = prepro.scale(D)
#     normS = prepro.scale(S)
    
    #===========================================================================
    # iterate until converge or max steps
    #===========================================================================
    print("start to train model...")
    for nStep in xrange(nMaxStep):
        currentU = np.copy(U)
        currentP = np.copy(P)
        currentV = np.copy(V)
        currentQ = np.copy(Q)
        
        # compute error
        mtCurrentErrorR, mtCurrentErrorD, mtCurrentErrorS, dCurrentRmseR, dCurrentRmseD, dCurrentRmseS, dCurrentLoss = \
          computeResidualError(R, normD, normS, currentU, currentV, currentP, currentQ, mu, weightR, weightD, weightS)
        
        # save RMSE
        if (lsTrainingRMSE is not None):
            dcRMSE = {}
            dcRMSE['rmseR'] = dCurrentRmseR
            dcRMSE['rmseD'] = dCurrentRmseD
            dcRMSE['rmseS'] = dCurrentRmseS
            dcRMSE['loss'] = dCurrentLoss
            lsTrainingRMSE.append(dcRMSE)
        
        # output
        if (bDebugInfo):
            print("------------------------------")
            print("step %d" % (nStep) )
            print("    RMSE(R) = %f" % dCurrentRmseR )
            print("    RMSE(D) = %f" % dCurrentRmseD )
            print("    RMSE(S) = %f" % dCurrentRmseS )
            print("    loss = %f" % dCurrentLoss )
            
            
        # compute partial gradient
        gradU, gradV, gradP, gradQ = \
            computeParitialGraident(mtCurrentErrorR, mtCurrentErrorD, mtCurrentErrorS, currentU, currentV, currentP, currentQ)
        
        #=======================================================================
        # search for max step
        #=======================================================================
        gamma = 1
        while(True):
#             print('--> trying gamma=%f...' % gamma)
            # U
            nextU = currentU - gamma*gradU
            # P
            nextP = currentP - gamma*gradP
            # V
            nextV = currentV - gamma*gradV
            # Q
            nextQ = currentQ - gamma*gradQ
             
            dNextErrorR, dNextErrorD, dNextErrorS, dNextRmseR, dNextRmseD, dNextRmseS, dNextLoss = \
                computeResidualError(R, normD, normS, nextU, nextV, nextP, nextQ, mu, weightR, weightD, weightS)
                 
            if (dNextLoss >= dCurrentLoss):
                # search for max step size
                gamma = gamma/2.0 
            else:
                # save the best update
                print('-->max gamma=%f' % gamma)
                U = nextU
                V = nextV
                P = nextP
                Q = nextQ
                break     
        
        #=======================================================================
        # check convergence
        #=======================================================================
        print('checking convergence...')
        dChange = dCurrentLoss-dNextLoss
        if( dChange>=0.0 and dChange< g_dEpsilon): 
            print("converged~! loss=%f, rmseR=%f" % (dNextLoss, dNextRmseR) )
            
            # save RMSE
            if (lsTrainingRMSE is not None):
                dcRMSE = {}
                dcRMSE['rmseR'] = dCurrentRmseR
                dcRMSE['rmseD'] = dCurrentRmseD
                dcRMSE['rmseS'] = dCurrentRmseS
                dcRMSE['loss'] = dCurrentLoss
                lsTrainingRMSE.append(dcRMSE)
                
            if (bDebugInfo):
                print("------------------------------")
                print("final step:")
                print("    RMSE(R) = %f" % dNextRmseR )
                print("    RMSE(normD) = %f" % dNextRmseD )
                print("    RMSE(normS) = %f" % dNextRmseS )
                print("    loss = %f" % dNextLoss )
            
            break
        
        elif(dChange < 0.0): # loss increases
            print "learning rate is too large, loss increases!"
            break
        else: # loss decreases, but is not converged
            pass 
        
#         #=======================================================================
#         # time to clean memory
#         #=======================================================================
#         print('time to release memory...')
#         gc.collect()
    #END
        
    #===========================================================================
    # test
    #===========================================================================
    print("start to test...")
#     predR_test =  (np.dot(U, V.T) + bu + bv ) + mu # use broadcast to add on each row/column
    predR_test =  np.dot(U, V.T) + mu
    _errorR_test = np.subtract(R, predR_test)
    errorR_test = np.multiply(weightR_test, _errorR_test)
    rmseR_test = np.sqrt( np.power(errorR_test, 2.0).sum() / (weightR_test==1.0).sum() )
    
    print("-->traning: %f, test: %f" % (lsTrainingRMSE[-1]['rmseR'], rmseR_test))
    
    return U, P, V, Q, mu, rmseR_test

def visualizeRMSETrend(lsRMSE):
    '''
        This function visualize the changing of RMSE
    '''
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2)
    df = pd.DataFrame(lsRMSE)
    df = df[['rmseD', 'rmseS', 'rmseR', 'loss']]
    df.columns = ['D', 'S', 'R', 'loss']
    ax0 = df[['D', 'S', 'R'] ].plot(ax=axes[0], style=['--','-.', '-' ], ylim=(0,1))
    ax0.set_xlabel('step')
    ax0.set_ylabel('RMSE')
    ax0.yaxis.set_ticks(np.arange(0.0, 1.0, 0.1))
    
    
    ax1 = df['loss'].plot(ax=axes[1], style='-*')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    
    plt.show()
    
if __name__ == '__main__':
    # setup
    arrAlphas = np.array([0.6, 0.2, 0.2])
    arrLambdas = np.array([1, 1, 1])
    f = 20
    dLearningRate = 0.001
    nMaxStep = 300
    
    # cross validation
    dMinRmseR = 100000.0
    lsBestTrainingRMSEs = None
    nFold = 5
    for i in xrange(nFold):
        
        # load data
        R = np.load('d:\\playground\\sh_xdr\\R_top500.npy')
        D = np.load('d:\\playground\\sh_xdr\\D_top500.npy')
        S = np.load('d:\\playground\\sh_xdr\\S_top500.npy')
        
        lsTrainingRMSE = []
    
        U, P, V, Q, mu, rmseR_test  = fit_test(R, D, S, arrAlphas, arrLambdas, f,\
                               dLearningRate, nMaxStep, lsTrainingRMSE, dTestRatio=min(1.0/nFold, 0.3), bDebugInfo=True)
        
        if (rmseR_test < dMinRmseR):
            dMinRmseR = rmseR_test
            lsBestTrainingRMSEs = lsTrainingRMSE
    
    print('====testCMF finished====\n-->best training rmse:%f, test rmse:%f' % (lsTrainingRMSE[-1]['rmseR'], dMinRmseR) ) 
    visualizeRMSETrend(lsBestTrainingRMSEs)
    print('finished')