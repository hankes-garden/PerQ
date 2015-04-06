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
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn import metrics


import matplotlib.pyplot as plt

g_dConvergenceThresold = 0.01
g_gamma0 = 0.1
g_power_t = 0.25


def getLearningRate(gamma, nIter):
    '''
        dynamically change learning rate w.r.t #iteration (using sklearn default: eta = eta0 / pow(t, g_power_t) )
    '''
    return gamma # use constant learning rate for demo

#     newGamma = g_gamma0 / pow(nIter+1, g_power_t)
#     return newGamma

#     return np.log2(nIter+1) / (nIter+1.0) # set to log(n)/n

def computeResidualError(R, D, S, U, V, P, Q, Bu, Bv, mu, Jm, Jn,\
                         weightR_train, weightR_test, weightD_train, weightS_train, \
                         arrAlphas, arrLambdas):
    #===========================================================================
    # compute initial error
    #===========================================================================
    # compute error in R (R = bu + U·V^T )
    predR =  np.dot(U, V.T) + np.dot(Bu, Jn.T)  + np.dot(Jm, Bv.T) + mu
#     predR =  np.dot(U, V.T)
    _errorR = np.subtract(R, predR)
    errorR_train = np.multiply(weightR_train, _errorR)
    
    # compute test error in R
    errorR_test = np.multiply(weightR_test, _errorR)
    
    # compute error in D
    predD = np.dot(U, P.T)
    _errorD = np.subtract(D, predD)
    errorD_train = np.multiply(weightD_train, _errorD)
    
    # compute error in S
    predS = np.dot(V, Q.T)
    _errorS = np.subtract(S, predS)
    errorS_train = np.multiply(weightS_train, _errorS)
    
    # compute rmse
    rmseR_train = np.sqrt( np.power(errorR_train, 2.0).sum() / weightR_train.sum() )
    rmseD_train = np.sqrt( np.power(errorD_train, 2.0).sum() / weightD_train.sum() )
    rmseS_train = np.sqrt( np.power(errorS_train, 2.0).sum() / weightS_train.sum() )
    rmseR_test = np.sqrt( np.power(errorR_test, 2.0).sum() / weightR_test.sum() )
    
    dTotalLost = (arrAlphas[0]/2.0) * np.power(np.linalg.norm(errorR_train, ord='fro'), 2.0) \
            + (arrAlphas[1]/2.0) * np.power(np.linalg.norm(errorD_train, ord='fro'), 2.0) \
            + (arrAlphas[2]/2.0) * np.power(np.linalg.norm(errorS_train, ord='fro'), 2.0) \
            + (arrLambdas[0]/2.0) * ( np.power(np.linalg.norm(U, ord='fro'), 2.0) + np.power(np.linalg.norm(V, ord='fro'), 2.0) ) \
            + (arrLambdas[1]/2.0) * np.power(np.linalg.norm(P, ord='fro'), 2.0) \
            + (arrLambdas[2]/2.0) * np.power(np.linalg.norm(Q, ord='fro'), 2.0) \
            + (arrLambdas[3]/2.0) * np.power(np.linalg.norm(Bu, ord='fro'), 2.0) \
            + (arrLambdas[4]/2.0) * np.power(np.linalg.norm(Bv, ord='fro'), 2.0)
            
            
    return errorR_train, errorD_train, errorS_train, rmseR_train, rmseR_test, rmseD_train, rmseS_train, dTotalLost

def computeParitialGraident(errorR, errorD, errorS, U, V, P, Q, Bu, Bv, Jm, Jn, arrAlphas, arrLambdas):
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
    
    # bu
    gradBu = ( -1.0 * arrAlphas[0] * np.dot(errorR, Jn) + arrLambdas[3]*Bu)
    
    # bv
    gradBv = ( -1.0 * arrAlphas[0] * np.dot(errorR.T, Jm) + arrLambdas[4]*Bv)
    
    return gradU, gradV, gradP, gradQ, gradBu, gradBv

def init(mtR, mtD, mtS, inplace, dReductionRatio=0.7):
    '''
        This function:
        1. return the weight matrices for R,D,S;
        2. fill missing value
        3. cast R to 0.0~1.0 if need
        4. feature scaling for D, S (R does not need feature scaling, 
           since all of its elements are in the same scale)
        5. reduce dimension of R and S 
           
        Note:
                1. if inplace=True, then the content of mtR, mtD, mtS will
                be modified (e.g., fill missing value with 0).
                2. in the returns of this function, zero is used for missing
                value, no more Nan
                
    '''
    
    #===========================================================================
    # copy data to prevent modification on original data
    # or don't copy to save memory
    #===========================================================================
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
    
    #===========================================================================
    # weight matrix for D, need to be done before filling nan
    #===========================================================================
    weightD = np.where(np.isnan(D), 0.0, 1.0)
    
    
    #===========================================================================
    # fill missing values in D and S with mean values as 
    # they will be feed to feature scaling
    #===========================================================================
    imp = prepro.Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
    D = imp.fit_transform(D)
    S = imp.fit_transform(S)

    #===========================================================================
    # feature scaling
    #===========================================================================
    print ('start to scale features...')
    # scaling features of D and S to [0,1]
    scaler = prepro.MinMaxScaler(copy=False)
    D = scaler.fit_transform(D)
    S = scaler.fit_transform(S)
    
    #===========================================================================
    # reduce video dimension
    #===========================================================================
    if (dReductionRatio < 1.0 ):
        print('start to reduce video dimension...')
        R, S = reduceVideoDimension(R, S, int(S.shape[0]*dReductionRatio))
        
        
    # filter out invalid tuple
    R_filtered, D_filtered, S_filtered = filterInvalidRecords(mtR, mtD, mtS)
        
    # get weight matrix
    arrMaskR = (np.isnan(R))
    weightR = np.where(arrMaskR, 0.0, 1.0) 
    weightS = np.where(np.isnan(S), 0.0, 1.0)
    
    
    # fill missing value in R
    R[np.isnan(R)] = 0.0
    
    return R, D, S, weightR, weightD, weightS

def fit(R, D, S, weightR_train, weightR_test, weightD_train, weightS_train, \
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
    
    # bu, bv
    Bu = np.random.rand(R.shape[0], 1)
    Bv = np.random.rand(R.shape[1], 1)
    
    # Jm, Jn
    Jm = np.ones((R.shape[0], 1), dtype=np.float64)
    Jn = np.ones((R.shape[1], 1), dtype=np.float64)
    
    # compute mu, must be calculated after masking test data
    mu = ( (R*weightR_train).sum()*1.0) / weightR_train.sum()
    
#     # compute bu ( for mean normalization), must be calculated after masking test data
#     # bu will only be computed if there are more than 2 tuples in the records of this user
#     bu = np.where(weightR_train.sum(axis=1)>=3.0, ( ( (R-mu)*weightR_train).sum(axis=1)*1.0) / weightR_train.sum(axis=1), 0.0)
#     bu = np.reshape(bu, (bu.shape[0], 1) )
#     bu[np.isnan(bu)] = 0.0
    
    dScale = 1000.0
    arrAlphas_scaled = arrAlphas.copy()
#     arrAlphas_scaled[0] = dScale * arrAlphas_scaled[0] / weightR_train.sum()
#     arrAlphas_scaled[1] = dScale * arrAlphas_scaled[1] / weightD_train.sum()
#     arrAlphas_scaled[2] = dScale * arrAlphas_scaled[2] / weightS_train.sum()
    
    # lambda will be tuned more easily if we do not scale it.
    arrLambdas_scaled = arrLambdas.copy()
#     arrLambdas_scaled[0] = dScale * arrLambdas_scaled[0] / min( (U.shape[0]*U.shape[1]), (V.shape[0]*V.shape[1]) )
#     arrLambdas_scaled[1] = dScale * arrLambdas_scaled[1] / (P.shape[0] * P.shape[1])
#     arrLambdas_scaled[2] = dScale * arrLambdas_scaled[2] / (Q.shape[0] * Q.shape[1])
    
    print "arrAlphas_scaled = ", arrAlphas_scaled
    print "arrLambdas_scaled = ", arrLambdas_scaled
    
    #===========================================================================
    # iterate until converge or max steps
    #===========================================================================
    for nStep in xrange(nMaxStep):
        currentU = U
        currentP = P
        currentV = V
        currentQ = Q
        currentBu = Bu
        currentBv = Bv
        
        # compute error
        mtCurrentErrorR, mtCurrentErrorD, mtCurrentErrorS, \
        dCurrentRmseR_train, dCurrentRmseR_test, dCurrentRmseD, dCurrentRmseS, \
        dCurrentLoss = computeResidualError(R, D, S, \
                                            currentU, currentV, currentP, currentQ, \
                                            currentBu, currentBv, mu, Jm, Jn,\
                                            weightR_train, weightR_test, weightD_train, weightS_train, \
                                            arrAlphas_scaled, arrLambdas_scaled)
        
        # save RMSE
        if (lsTrainingTrace is not None):
            dcRMSE = {}
            dcRMSE['rmseR'] = dCurrentRmseR_train
            dcRMSE['rmseR_test'] = dCurrentRmseR_test
            dcRMSE['rmseD'] = dCurrentRmseD
            dcRMSE['rmseS'] = dCurrentRmseS
            dcRMSE['loss'] = dCurrentLoss
            lsTrainingTrace.append(dcRMSE)
        
        # output
        if (bDebugInfo):
            print("------------------------------")
            print("step %d" % (nStep) )
            print("    RMSE(R) = %f" % dCurrentRmseR_train )
            print("    RMSE(R)_test = %f" % dCurrentRmseR_test )
            print("    RMSE(D) = %f" % dCurrentRmseD )
            print("    RMSE(S) = %f" % dCurrentRmseS )
            print("    loss = %f" % dCurrentLoss )
            
            
        # compute partial gradient
        gradU, gradV, gradP, gradQ, gradBu, gradBv = \
            computeParitialGraident(mtCurrentErrorR, mtCurrentErrorD, mtCurrentErrorS, \
                                    currentU, currentV, currentP, currentQ, \
                                    currentBu, currentBv, Jm, Jn, \
                                    arrAlphas_scaled, arrLambdas_scaled)
        
        #=======================================================================
        # search for max step
        #=======================================================================
        dNextRmseR_train = None
        dNextRmseR_test = None
        dNextRmseD = None
        dNextRmseS = None
        dNextLoss = None
        gamma = g_gamma0
        while(True):
            # try a possible step
            nextU = currentU - gamma*gradU
            nextP = currentP - gamma*gradP
            nextV = currentV - gamma*gradV
            nextQ = currentQ - gamma*gradQ
            nextBu = currentBu - gamma*gradBu
            nextBv = currentBv - gamma*gradBv
             
            mtNextErrorR, mtNextErrorD, mtNextErrorS, \
            dNextRmseR_train, dNextRmseR_test, dNextRmseD, dNextRmseS, \
            dNextLoss = computeResidualError(R, D, S, nextU, nextV, nextP, nextQ, \
                                             nextBu, nextBv, mu, Jm, Jn,\
                                             weightR_train, weightR_test, weightD_train, weightS_train, \
                                             arrAlphas_scaled, arrLambdas_scaled)
                 
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
                Bu = nextBu
                Bv = nextBv
                break     
        
        #=======================================================================
        # check convergence
        #=======================================================================
        dChange = dCurrentLoss-dNextLoss
        if( dChange>=0.0 and dChange<=g_dConvergenceThresold): 
            print("converged @ step %d: change:%f, loss=%f, rmseR_test=%f" % (nStep, dChange, dNextLoss, dNextRmseR_test) )
            
            # save RMSE
            if (lsTrainingTrace is not None):
                dcRMSE = {}
                dcRMSE['rmseR'] = dNextRmseR_train
                dcRMSE['rmseR_test'] = dNextRmseR_test
                dcRMSE['rmseD'] = dNextRmseD
                dcRMSE['rmseS'] = dNextRmseS
                dcRMSE['loss'] = dNextLoss
                lsTrainingTrace.append(dcRMSE)
                
            if (bDebugInfo):
                print("------------------------------")
                print("final step:")
                print("    RMSE(R) = %f" % dNextRmseR_train )
                print("    RMSE(R)_test = %f" % dNextRmseR_test )
                print("    RMSE(normD) = %f" % dNextRmseD )
                print("    RMSE(normS) = %f" % dNextRmseS )
                print("    loss = %f" % dNextLoss )
            
            break
        
        elif(dChange < 0.0): # loss increases
            print "learning rate is too large, loss increases!"
            break
        
        else: # loss decreases, but is not converged, do nothing
            if(bDebugInfo):
                print("-->change: %f" % dChange)
            pass 
        
    #END step
    
    
    return U, V, P, Q, Bu, Bv, mu, Jm, Jn

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

def crossValidate(R, D, S, weightR, weightD, weightS, \
                  arrAlphas, arrLambdas, f, nMaxStep, nFold=10, \
                  bDebugInfo=False):
    '''
        This function cross-validates collective matrix factorization model. 
        In particular, it perform:
        1. cut the data into folds
        2. fit
        3. test
        
        params:
            mtR         - user-video matrix, m-by-n sparse matrix, 
                          each element is a unsigned integer which belong to 0~100, 0 for unknown
            mtD         - user-profile matrix, m-by-l (dense, 0 for unknown)
            mtS         - video-quality matrix, n-by-h (dense, 0 for unknown)
            arrAlphas   - re-construction weights for R, D, S
            arrLambdas  - lambdas for regularization of (U, V), P, Q
            f           - number of latent factors
            nMaxStep    - max iteration steps
            nFold       - number of folds to validate
            
        return:
            dcResult - train/test result for R of each fold
            lsBestTrainingTrace - RMSE of each step in best fold
        
        Note:
            1. this function might change the content of input matrices (depends on init() function);
            2. input matrices may use both np.nan and 255 to represent missing values, 
               but in the computing core of CMF, we use 0 to represent missing value;
    '''
    
    #===========================================================================
    # cross validation
    #===========================================================================
    print('start cross validation...')
    
    # cut
    arrNonzeroRows, arrNonzeroCols = np.nonzero(R) # Note, we have already filled missing value in R by 0
    kf = cross_validation.KFold(len(arrNonzeroRows), nFold, shuffle=True)

    dcResults = {}
    nCount = 0
    dBestRmseR_test = 9999999999.0
    lsBestTrainingTrace = None
    
    for arrTrainIndex, arrTestIndex in kf:
        print("%d-th of %d folds..." % (nCount, nFold) )
        
        #=======================================================================
        # prepare train/test data    
        #=======================================================================
        weightR_train = np.copy(weightR)
        
        # don't use these selected elements to train
        weightR_train[arrNonzeroRows[arrTestIndex], arrNonzeroCols[arrTestIndex]] = 0.0
        
        # set weight R for testing
        weightR_test = np.zeros(weightR_train.shape)
        weightR_test[arrNonzeroRows[arrTestIndex], arrNonzeroCols[arrTestIndex]] = 1.0
        
        # TODO: will it be a problem if I do not mask corresponding tuples in D and S?
        
        #=======================================================================
        # train
        #=======================================================================
        lsTrainingTrace = []
        U, V, P, Q, \
        Bu, Bv, mu, Jm, Jn = fit(R, D, S, weightR_train, weightR_test, weightD, weightS, \
                             f, arrAlphas, arrLambdas, nMaxStep, \
                             lsTrainingTrace, bDebugInfo)
        
        #===========================================================================
        # test
        #===========================================================================
        predR_test =  np.dot(U, V.T) + np.dot(Bu, Jn.T)  + np.dot(Jm, Bv.T) + mu
#         predR_test =  np.dot(U, V.T)
        _errorR_test = np.subtract(R, predR_test)
        errorR_test = np.multiply(weightR_test, _errorR_test)
        rmseR_test = np.sqrt( np.power(errorR_test, 2.0).sum() / weightR_test.sum() )
        maeR_test = (np.abs(errorR_test)).sum() / weightR_test.sum()
        
       
        # save fold result
        dcResults[nCount] = {'train':lsTrainingTrace[-1]['rmseR'], 'test':rmseR_test, 'mae':maeR_test}
        
        if (rmseR_test < dBestRmseR_test):
            dBestRmseR_test = rmseR_test
            lsBestTrainingTrace = lsTrainingTrace
        
        # print out 
        print("rmse_traning=%f, rmse_test=%f, mae_test=%f" % 
              (lsTrainingTrace[-1]['rmseR'], rmseR_test, maeR_test))
        
        if(bDebugInfo is True):
            visualizeRMSETrend(lsTrainingTrace)
            
#         # investigate reason
#         dfErrorReason = InvestigateError(errorR_test, R, predR_test, mu, Bu, Bv, U, V, weightR_test)
#         
#         dfErrorReason.to_csv("d:\\playground\\personal_qoe\\error\\dfErrorReason_%d_%f.csv" % (nCount, rmseR_test), \
#                      columns=['pos','error','R','predR','bu','bv','uv', 'arr_nz_row','num_test_row', 'arr_nz_col','num_test_col'])

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
    df = df[['rmseD', 'rmseS', 'rmseR', 'rmseR_test', 'loss']]
    df.columns = ['D', 'S', 'R', 'R_test', 'loss']
    ax0 = df[['D', 'S', 'R', 'R_test'] ].plot(ax=axes[0], style=['--','-.', '-', '-+' ], ylim=(0,1))
    ax0.set_xlabel('step')
    ax0.set_ylabel('RMSE')
    ax0.yaxis.set_ticks(np.arange(0.0, 1.0, 0.1))
    
    ax1 = df['loss'].plot(ax=axes[1], style='-*')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    
    plt.show()
    
def reduceVideoDimension(mtR, mtS, nTargetDimension):
    '''
        This function first cluster videos based on their features,
        then merge similar videos which belongs to same cluster into
        one video.  
        
        parameters:
                    mtR - NAN for missing value 
                    mtS - no missing valued any more
                    nTargetDimension - number of dimension to achieve
                    
        returns:
                    mtR_merged - reduced R matrix, NAN for missing values
                    mtS_merged - reduced S matrix, no more missing value
        Note:
                    this function should be called after D,S are normalized
                    and missing values are also filled, but before filling 
                    the missing values of R.
    '''
    #===========================================================================
    # clustering
    #===========================================================================
    print('start to clustering similar videos...')
    aggCluster = AgglomerativeClustering(n_clusters=nTargetDimension, linkage='complete', \
                                         affinity='euclidean')
    arrClusterIDs = aggCluster.fit_predict(mtS)
   
    #===========================================================================
    # merge
    #===========================================================================
    print('start to merge clusters...')
    srClusterIDs = pd.Series(arrClusterIDs)
    mtS_merged = None
    mtR_merged = None
    for cid in xrange(nTargetDimension):
        lsVideos2Merge = srClusterIDs[srClusterIDs==cid].index.tolist()
        
        #merge S
        arrMergedVideoFeatures = np.nanmean(mtS[lsVideos2Merge, :], axis=0, keepdims=True)
        if (mtS_merged is None):
            mtS_merged = arrMergedVideoFeatures
        else:
            mtS_merged = np.vstack([mtS_merged, arrMergedVideoFeatures])
            
        #merge R
        arrMergedVideoRatio = np.nanmean(mtR[:, lsVideos2Merge], axis=1, keepdims=True)
        if (mtR_merged is None):
            mtR_merged = arrMergedVideoRatio
        else:
            mtR_merged = np.hstack([mtR_merged, arrMergedVideoRatio])
        
    return mtR_merged, mtS_merged


def filterInvalidRecords(mtR, mtD, mtS):
    '''
        This function filter out invalid tuples in a cascade way
        
        Note: Nan for missing value
    '''
    R = mtR.copy()
    D = mtD.copy()
    S = mtS.copy()
    
    # criteria of invalid tuples
    mtInvalidMask = (R<0.1)
    
    # mark those tuples as invalid ones
    R[mtInvalidMask] = np.nan
    
    # reduce trivial rows & columns
    mtValidMask = ~np.isnan(R)
    
    # find out users whose does not have any valid video record 
    arrValidCount_user = np.sum(mtValidMask, axis=1)
    srValidCount_user = pd.Series(arrValidCount_user)
    lsTrivialUsers = srValidCount_user[srValidCount_user==0].index
    
    # find out videos which has not been watched validly
    arrValidCount_video = np.sum(mtValidMask, axis=0)
    srValidCount_video = pd.Series(arrValidCount_video)
    lsTrivialVideos = srValidCount_video[srValidCount_video==0].index
    
    # delete
    R = np.delete(R, lsTrivialUsers, axis=0)
    D = np.delete(D, lsTrivialUsers, axis=0)
    
    R = np.delete(R, lsTrivialVideos, axis=1)
    S = np.delete(S, lsTrivialVideos, axis=0)
    
    return R, D, S
            

def InvestigateError(matErrorR_test, R, predR_test, mu, Bu, Bv, U, V, weightR_test):
    '''
        This function investigate why there is such a large error
    '''
    # transform into dataframe
    arrNZRows, arrNZCols = matErrorR_test.nonzero()
    lsData = []
    for tp in zip(arrNZRows, arrNZCols):
        val = matErrorR_test[tp[0], tp[1]]
        lsData.append({'pos':tp, 'error':val, 'R':0.0, 'predR':0.0, 'bu':0.0, 'bv':0.0, 'uv':0.0, \
                       'arr_nz_row':0, 'num_test_row':0, \
                       'arr_nz_col':0, 'num_test_col':0 })
    
    # iterate over all errors, find out why
    for row in lsData:
        nRowID = row['pos'][0]
        nColID = row['pos'][1]
        row['R'] = R[nRowID, nColID]
        row['predR'] = predR_test[nRowID, nColID]
        row['bu'] = Bu[nRowID][0]
        row['bv'] = Bv[nColID][0]
        row['uv'] = np.dot(U, V.T)[nRowID, nColID]
        row['arr_nz_row'] = R[nRowID, R[nRowID,:].nonzero() ]
        row['num_test_row'] = weightR_test[nRowID, :].sum()
        row['arr_nz_col'] = R[R[:, nColID].nonzero(), nColID]
        row['num_test_col'] = weightR_test[:, nColID].sum()
    
    dfErrorReason = pd.DataFrame(lsData)
    
    return dfErrorReason

def testCMF(**kwargs):
    # load data
    mtR = np.load('d:\\playground\\personal_qoe\\data\\sh\\mtR_0discre_rand1000.npy')
    mtD = np.load('d:\\playground\\personal_qoe\\data\\sh\\mtD_0discre_rand1000.npy')
    mtS = np.load('d:\\playground\\personal_qoe\\data\\sh\\mtS_0discre_rand1000.npy')
    
    # setup parameter
    arrAlphas = kwargs['alphas'] # will be scaled in the core of CMF
    arrLambdas = kwargs['lambdas'] # U&V, P, Q, Bu, Bv
    f = kwargs['f']
    nMaxStep = kwargs['max_step']
    nFold = kwargs['folds']
    bDebugTrace = kwargs['debug_trace']
    dReductionRatio = kwargs['video_reduction_ratio']
    bVisualize = kwargs['visualize']
     
    # filter out invalid tuple
    R_filtered, D_filtered, S_filtered = filterInvalidRecords(mtR, mtD, mtS)
    
    # init (prepare weight matrix, scale features and aggregate videos)
    R_reduced, D_reduced, S_reduced, \
    weightR_reduced, weightD_reduced, weightS_reduced = init(R_filtered, D_filtered, S_filtered, inplace=False,
                                                             dReductionRatio=dReductionRatio)
    if (bDebugTrace is True):
        print "R_reduced.shape=", R_reduced.shape
        print "D_reduced.shape=", D_reduced.shape
        print "S_reduced.shape=", S_reduced.shape
        print "weightR_reduced.sum=", weightR_reduced.sum()
        print "weightD_reduced.sum=", weightD_reduced.sum()
        print "weightS_reduced.sum=", weightS_reduced.sum()
    
    # cross validation
    dcResult, lsBestTrainingRMSEs = crossValidate(R_reduced, D_reduced, S_reduced, \
                                                  weightR_reduced, weightD_reduced, weightS_reduced, \
                                                  arrAlphas, arrLambdas, \
                                                  f, nMaxStep, nFold, \
                                                  bDebugTrace)

    # output result
    for k, v in dcResult.items():
        print("fold %d: train=%f, test=%f, mae=%f" % (k, v['train'], v['test'], v['mae']) )
    
    lsTestRMSE = [v['test'] for v in dcResult.values() ] 
    dMean_rmse = np.mean(lsTestRMSE)
    dStd_rmse = np.std(lsTestRMSE)
    print('-->RMSE: mean=%f, std=%f' % (dMean_rmse, dStd_rmse))
    
    lsTestMAE = [v['mae'] for v in dcResult.values() ] 
    dMean_mae = np.mean(lsTestMAE)
    dStd_mae = np.std(lsTestMAE)
    print('-->MAE: mean=%f, std=%f' % (dMean_mae, dStd_mae))
    
    # visualize
    if (bVisualize is True):
        visualizeRMSETrend(lsBestTrainingRMSEs)
    
    print('finished*********')
    
    return dMean_rmse, dStd_rmse, dMean_mae, dStd_mae

def investigateImpactOfParameters(strParamName, bPlot, strPath=None):
    '''
        This function investigate the impact of parameter on model performance by
        trying different combinations of parameters
        
        Parameters:
            strParamName - parameter to examine
            bPlot        - visualize the result if it is true
            strPath      - save the result in csv format if it is true
    '''
    
    # set default param
    dcTestParam = {}
    dcTestParam['alphas'] = np.array([1.0, 0.05, 0.12]) # will NOT be scaled in the core of CMF!
    dcTestParam['lambdas'] = np.array([0.9, 0.9, 0.9, 0.9, 0.9]) # U&V, P, Q, Bu, Bv
    dcTestParam['f'] = 10
    dcTestParam['video_reduction_ratio'] = 0.5
    dcTestParam['max_step'] = 500
    dcTestParam['folds'] = 5
    
    dcTestParam['debug_trace'] = False
    dcTestParam['visualize'] = False
    
    #===========================================================================
    # try different params
    #===========================================================================
    print("investigating impact of %s..." % (strParamName) )
    lsResults = []
    if (strParamName == 'c'):
        for i in range(1,10,1):
            dcTestParam['video_reduction_ratio'] = i * 0.1
            dMean_rmse, dStd_rmse, dMean_mae, dStd_mae = testCMF(**dcTestParam)
            lsResults.append({strParamName:i*0.1, 'rmse_mean':dMean_rmse, 'rmse_std':dStd_rmse, 'mae_mean':dMean_mae, 'mae_std':dStd_mae} )
        
    elif (strParamName == 'f'):
        for i in range(5, 25, 5):
            dcTestParam['f'] = i
            dMean_rmse, dStd_rmse, dMean_mae, dStd_mae = testCMF(**dcTestParam)
            lsResults.append({strParamName:i, 'rmse_mean':dMean_rmse, 'rmse_std':dStd_rmse, 'mae_mean':dMean_mae, 'mae_std':dStd_mae} )
        
    elif(strParamName == 'alpha_2'):
        dVar = 0.01
        for i in range(0, 5, 1):
            dcTestParam['alphas'] = np.array([1.0, dVar, 0.12])
            dMean_rmse, dStd_rmse, dMean_mae, dStd_mae = testCMF(**dcTestParam)
            lsResults.append({strParamName:dVar, 'rmse_mean':dMean_rmse, 'rmse_std':dStd_rmse, 'mae_mean':dMean_mae, 'mae_std':dStd_mae} )
            dVar = dVar * 5
    
    elif(strParamName == 'alpha_3'):
        dVar = 0.0001
        for i in range(0, 5, 1):
            dcTestParam['alphas'] = np.array([1.0, 0.1, dVar])
            dMean_rmse, dStd_rmse, dMean_mae, dStd_mae = testCMF(**dcTestParam)
            lsResults.append({strParamName:dVar, 'rmse_mean':dMean_rmse, 'rmse_std':dStd_rmse, 'mae_mean':dMean_mae, 'mae_std':dStd_mae} )
            dVar = dVar * 10.0
    
    elif(strParamName == 'lambda'):
            dVar = 0.1
            for i in range(0, 5, 1):
                dcTestParam['lambdas'] = np.array([dVar] *5)
                dMean_rmse, dStd_rmse, dMean_mae, dStd_mae = testCMF(**dcTestParam)
                lsResults.append({strParamName:dVar, 'rmse_mean':dMean_rmse, 'rmse_std':dStd_rmse, 'mae_mean':dMean_mae, 'mae_std':dStd_mae} )
                dVar = dVar * 3

    else:
        print("Error: unknown parameter name %s." % strParamName)
        return
    
    #===========================================================================
    # save result
    #===========================================================================
    if(strPath is not None):
            # construct data frame
            df = pd.DataFrame(lsResults)
            df.sort(strParamName, ascending=True, inplace=True)
            df.set_index(strParamName, inplace=True)
            df.to_csv( "%s%s.csv" % (strPath, strParamName) )
            
    #===========================================================================
    # draw error bar
    #===========================================================================

    if(bPlot is True):
        drawImapctOfParameter(lsResults, strParamName)
    
    for ret in lsResults:
        print ret
    print("Investigation finished.")
        
   
def drawImapctOfParameter(lsResult, strParamName):
    # construct data frame
    df = pd.DataFrame(lsResult)
    df.sort(strParamName, ascending=True, inplace=True)
    df.set_index(strParamName, inplace=True)
    
    # plot
    ax0 = plt.figure().add_subplot(111)
    ax0.errorbar(x=df.index.tolist(), y=df['mae_mean'].tolist(), yerr=df['mae_std'].tolist(), marker='s', lw=2, label='mae')
    ax0.errorbar(x=df.index.tolist(), y=df['rmse_mean'], yerr=df['rmse_std'], marker='o', label='rmse')

    # set legend
    handles, labels = ax0.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax0.legend(handles, labels, loc='best', numpoints=1)

    # set axis
    ax0.yaxis.set_ticks(np.arange(0.0, 1.5, 0.1))
    ax0.set_xlabel(strParamName)
    ax0.set_ylabel('error rate')
    
    plt.show()
    
if __name__ == '__main__':
    investigateImpactOfParameters('f', False, 'D:\\playground\\personal_qoe\\result\\impact_of_param\\')
    investigateImpactOfParameters('lambda', False, 'D:\\playground\\personal_qoe\\result\\impact_of_param\\')
    investigateImpactOfParameters('c', False, 'D:\\playground\\personal_qoe\\result\\impact_of_param\\')
    investigateImpactOfParameters('alpha_3', False, 'D:\\playground\\personal_qoe\\result\\impact_of_param\\')