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
from sklearn import cross_validation
import gc


g_dConvergenceThresold = 0.1
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

def computeResidualError(R, D, S, U, V, P, Q, mu, weightR, weightD, weightS, arrAlphas, arrLambdas):
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

def computeParitialGraident(errorR, errorD, errorS, U, V, P, Q, arrAlphas, arrLambdas):
    # U
    gradU = ( -1.0*arrAlphas[0]*np.dot(errorR, V) - arrAlphas[1]*np.dot(errorD, P) \
                      + arrLambdas[0]*U )
    # P
    gradP = ( -1.0*arrAlphas[1]*np.dot(errorD.T, U) + arrLambdas[1]*P )
    
    # V
    gradV = ( -1.0*arrAlphas[0]*np.dot(errorR.T, U) - arrAlphas[2]*np.dot(errorS, Q) \
                      + arrLambdas[0]*V )
    # Q
    gradQ = ( -1.0 * arrAlphas[2]*np.dot(errorS.T, V) + arrLambdas[2]*Q )
    
    return gradU, gradV, gradP, gradQ

def init(mtR, mtD, mtS, inplace, missing_value, bCastR):
    '''
        This function:
        1. return the weight matrices for R,D,S;
        2. fill missing value
        3. cast R to 0.0~1.0
        4. feature scaling for D, S (R does not need feature scaling, 
           since all of its elements are in the same scale)
           
        Note:
                if inplace=True, then the content of mtR, mtD, mtS will
                be modified (e.g., fill missing value with 0)
    '''
    
    # copy data to prevent modification on original data
    # or don't copy to save memory
    R = None
    D = None
    S = None
    
    if (inplace):
        R = mtR
        D = mtD
        S = mtS
    else:
        R = np.copy(mtR)
        D = np.copy(mtD)
        S = np.copy(mtS)
        
    print('start to construct weight matrices...')
    # weight matrix for sparse matrices, need to be done before filling nan
    weightR = None
    if (missing_value is not None):
        weightR = np.where(R==missing_value, 0.0, 1.0)
    else:
        weightR = np.where(np.isnan(R), 0.0, 1.0)
        
    weightD = np.where(np.isnan(D), 0.0, 1.0)
    weightS = np.where(np.isnan(S), 0.0, 1.0)
    
    # fill missing values with 0
    if (missing_value is not None):
        R[R==missing_value] = 0
    else:
        R[np.isnan(R)] = 0
    D[np.isnan(D)] = 0.0
    S[np.isnan(S)] = 0.0
    
    #===========================================================================
    # feature scaling
    #===========================================================================
    # cast R into [0.0, 1.0]
    if(bCastR):
        R = (R *1.0/100.0)
    
    # scaling features to [0,1]
    print ('start to scale features...')
    min_max_scaler = prepro.MinMaxScaler(copy=False)
    normD = min_max_scaler.fit_transform(D)
    normS = min_max_scaler.fit_transform(S)
    
    return R, D, S, weightR, weightD, weightS, normD, normS

def fit(R, D, S, weightR, weightD, weightS, \
        f, arrAlphas, arrLambdas, nMaxStep, \
        lsTrainingTrace, bDebugInfo=True):
    '''
        This function train CMF based on given input and params
    '''
    #===========================================================================
    # init low rank matrices
    #===========================================================================
    # D = U·P^T
    U = np.random.rand(D.shape[0], f)
    P = np.random.rand(D.shape[1], f)
    
    # S = V·Q^T
    V = np.random.rand(S.shape[0], f)
    Q = np.random.rand(S.shape[1], f)
    
    # compute mu ( for mean normalization), must be calculated after masking test data
    mu = ( (R*weightR).sum(axis=1)*1.0) / weightR.sum(axis=1)
    mu = np.reshape(mu, (mu.shape[0], 1) )
    mu[np.isnan(mu)] = 0.0
    
    #===========================================================================
    # iterate until converge or max steps
    #===========================================================================
    print("start to train model...")
    for nStep in xrange(nMaxStep):
        currentU = U
        currentP = P
        currentV = V
        currentQ = Q
        
        # compute error
        mtCurrentErrorR, mtCurrentErrorD, mtCurrentErrorS, \
        dCurrentRmseR, dCurrentRmseD, dCurrentRmseS, \
        dCurrentLoss = computeResidualError(R, D, S, currentU, currentV, currentP, currentQ, mu, \
                                            weightR, weightD, weightS, arrAlphas, arrLambdas)
        
        dNextLoss = None
        dNextRmseR = None
        
        # save RMSE
        if (lsTrainingTrace is not None):
            dcRMSE = {}
            dcRMSE['rmseR'] = dCurrentRmseR
            dcRMSE['rmseD'] = dCurrentRmseD
            dcRMSE['rmseS'] = dCurrentRmseS
            dcRMSE['loss'] = dCurrentLoss
            lsTrainingTrace.append(dcRMSE)
        
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
            computeParitialGraident(mtCurrentErrorR, mtCurrentErrorD, mtCurrentErrorS, \
                                    currentU, currentV, currentP, currentQ, arrAlphas, arrLambdas)
        
        #=======================================================================
        # search for max step
        #=======================================================================
        gamma = 1.0
        while(True):
            # U
            nextU = currentU - gamma*gradU
            # P
            nextP = currentP - gamma*gradP
            # V
            nextV = currentV - gamma*gradV
            # Q
            nextQ = currentQ - gamma*gradQ
             
            dNextErrorR, dNextErrorD, dNextErrorS, dNextRmseR, dNextRmseD, dNextRmseS, dNextLoss = \
                computeResidualError(R, D, S, nextU, nextV, nextP, nextQ, mu, \
                                     weightR, weightD, weightS, arrAlphas, arrLambdas)
                 
            if (dNextLoss >= dCurrentLoss):
                # search for max step size
                gamma = gamma/2.0 
            else:
                # save the best update
                if (bDebugInfo):
                    print('-->max gamma=%f' % gamma)
                U = nextU
                V = nextV
                P = nextP
                Q = nextQ
                break     
        
        #=======================================================================
        # check convergence
        #=======================================================================
        dChange = dCurrentLoss-dNextLoss
        if( dChange>=0.0 and dChange<=g_dConvergenceThresold): 
            print("converged~! loss=%f, rmseR=%f" % (dNextLoss, dNextRmseR) )
            
            # save RMSE
            if (lsTrainingTrace is not None):
                dcRMSE = {}
                dcRMSE['rmseR'] = dCurrentRmseR
                dcRMSE['rmseD'] = dCurrentRmseD
                dcRMSE['rmseS'] = dCurrentRmseS
                dcRMSE['loss'] = dCurrentLoss
                lsTrainingTrace.append(dcRMSE)
                
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
        
        else: # loss decreases, but is not converged, do nothing
            pass 
        
    #END step
    
    
    return U, V, P, Q, mu

def pred(R, D, S, U, V, P, Q, mu, weightR_test):
    #===========================================================================
    # test
    #===========================================================================
    print("start to test...")
    predR_test =  np.dot(U, V.T) + mu
    _errorR_test = np.subtract(R, predR_test)
    errorR_test = np.multiply(weightR_test, _errorR_test)
    rmseR_test = np.sqrt( np.power(errorR_test, 2.0).sum() / (weightR_test==1.0).sum() )
    return rmseR_test

def cross_validate(mtR, mtD, mtS, arrAlphas, arrLambdas, f, nMaxStep, nFold=10, \
                   missing_value=None, bCastR = False, inplace=False, bDebugInfo=False):
    '''
        This function cross-validates collective matrix factorization model. 
        In particular, it perform:
        1. initialize 
        2. fit
        3. test
        
        
        params:
            mtR         - user-video matrix, m-by-n sparse matrix, 
                          each element is a unsigned integer which belong to 0~100, 255 for unknown
            mtD         - user-profile matrix, m-by-l (dense, np.nan for unknown)
            mtS         - video-quality matrix, n-by-h (dense, np.nan for unknown)
            arrAlphas   - re-construction weights for R, D, S
            arrLambdas  - lambdas for regularization of (U, V), P, Q
            f           - number of latent factors
            nMaxStep    - max iteration steps
            nFold       - number of foldss
            
        return:
            dcResult - train/test rmse for R of each fold
            lsBestTrainingTrace - RMSE of each step in best fold
        
        Note:
            1. this function might change the content of input matrices (depends on init() function);
            2. input matrices may use both np.nan and 255 to represent missing values, 
               but in the computing core of CMF, we use 0 to represent missing value;
    '''
    
    print('start to initialize...')
    #===========================================================================
    # init
    #===========================================================================
    R, D, S, weightR, weightD, weightS, normD, normS = init(mtR, mtD, mtS, inplace, missing_value, bCastR)
    
    #===========================================================================
    # cross validation
    #===========================================================================
    print('start cross validation...')
    
    # cut
    arrNonzeroRows, arrNonzeroCols = np.nonzero(R)
    kf = cross_validation.KFold(len(arrNonzeroRows), nFold, shuffle=True)

    dcResults = {}
    nCount = 0
    dBestRmseR_test = 9999999999.0
    lsBestTrainingTrace = None
    for arrTrainIndex, arrTestIndex in kf:
        print("=================")
        print("%d of %d folds..." % (nCount, nFold) )
        #=======================================================================
        # prepare train/test data    
        #=======================================================================
        weightR_train = np.copy(weightR)
        
        # don't use these selected elements to train
        weightR_train[arrNonzeroRows[arrTestIndex], arrNonzeroCols[arrTestIndex]] = 0.0
        
        # set weight R for testing
        weightR_test = np.zeros(weightR_train.shape)
        weightR_test[arrNonzeroRows[arrTestIndex], arrNonzeroCols[arrTestIndex]] = 1.0
        
        #=======================================================================
        # train
        #=======================================================================
        lsTrainingTrace = []
        U, V, P, Q, mu = fit(R, normD, normS, weightR_train, weightD, weightS, \
                             f, arrAlphas, arrLambdas, nMaxStep, \
                             lsTrainingTrace, bDebugInfo)
        
        #===========================================================================
        # test
        #===========================================================================
        print("start to test...")
        predR_test =  np.dot(U, V.T) + mu
        _errorR_test = np.subtract(R, predR_test)
        errorR_test = np.multiply(weightR_test, _errorR_test)
        rmseR_test = np.sqrt( np.power(errorR_test, 2.0).sum() / weightR_test.sum() )
        
        dcResults[nCount] = {'train':lsTrainingTrace[-1]['rmseR'], 'test':rmseR_test}
        print("-->traning=%f, test=%f" % (lsTrainingTrace[-1]['rmseR'], rmseR_test))
        
        if (rmseR_test < dBestRmseR_test):
            dBestRmseR_test = rmseR_test
            lsBestTrainingTrace = lsTrainingTrace
        nCount += 1
    
    print("cross validation is finished: best train=%f, test=%f" % (lsBestTrainingTrace[-1]['rmseR'], dBestRmseR_test) )
    return dcResults, lsBestTrainingTrace

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
    # load data
    mtR = np.load('d:\\playground\\personal_qoe\\sh\\mtR_0discre_top100.npy')
    mtD = np.load('d:\\playground\\personal_qoe\\sh\\mtD_0discre_top100.npy')
    mtS = np.load('d:\\playground\\personal_qoe\\sh\\mtS_0discre_top100.npy')
    
    # setup
    arrAlphas = np.array([0.7, 0.2, 0.3])
    arrLambdas = np.array([0.5, 0.5, 0.5])
    f = 5
    nMaxStep = 300
    nFold = 5
     
    # cross validation
    dcResult, lsBestTrainingRMSEs = cross_validate(mtR, mtD, mtS, \
                                                   arrAlphas, arrLambdas, \
                                                   f, nMaxStep, nFold, \
                                                   missing_value=None, bCastR=False, \
                                                   inplace=False, bDebugInfo=False)
    # visualize
    visualizeRMSETrend(lsBestTrainingRMSEs)
    
    for k, v in dcResult.items():
        print("fold %d: train=%f, test=%f" % (k, v['train'], v['test']) )
    
    lsTestRMSE = [v['test'] for v in dcResult.values() ] 
    dMean = np.mean(lsTestRMSE)
    dStd = np.std(lsTestRMSE)
    print('mean=%f, std=%f' % (dMean, dStd))
    
    print('finished*********')