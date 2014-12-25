# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''

import numpy as np
import cmf.cmf_sgd as cmf
import data_processing.data2matrix as dm

def train(strDataPath):
    
    #===========================================================================
    # prepare matrix
    #===========================================================================
    if (strDataPath.endswith("/") is not True):
        strDataPath.append("/")
        
    D = np.load(strDataPath+"D.npy")
    S = np.load(strDataPath+"S.npy")
    R = np.load(strDataPath+"R.npy")
    
    #===========================================================================
    # fit
    #===========================================================================
    arrAlphas = np.array([0.7, 0.0, 0.3]) # R, D, S
    arrLambdas = np.array([0.4, 0.3, 0.3]) # R, D, S
    f = 5
    dLearningRate = 0.05
    nMaxStep = 100
    lsRMSE=[]

    bu, bv, U, P, V, Q  = cmf.fit(R, D, S, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsRMSE)
    
    cmf.visualizeRMSETrend(lsRMSE)
    
def multipleTrial(strInPath, lsParamSets, strParamName, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, dTestRatio):
    #===========================================================================
    # load data & transform into matrix
    #===========================================================================
    R, D, S = dm.transformNJData(strInPath)
    
    dcResult = {}
    nTrials = len(lsParamSets)
    for i in xrange(0, nTrials):
        lsRMSE = []
        #======================================================================
        # prepare params 
        #======================================================================
        if (strParamName == 'alpha'):
            arrAlphas = lsParamSets[i]
        elif (strParamName == 'lambda'):
            arrLambdas = lsParamSets[i] 
        elif (strParamName == 'f'):
            f = lsParamSets[i]
        else:
            print("Unknown param name!")
            return
         
        #======================================================================
        # trial
        #======================================================================
        print("---------------------------------------------------------------------")
        print "arrAlphas", arrAlphas
        print 'arrlambdas', arrLambdas
        print 'f', f
        print("---------------------------------------------------------------------")
        bu, bv, U, P, V, Q, rmseR  = cmf.fit(R, D, S, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsRMSE, dTestRatio )
        
        if type(lsParamSets[i]) is np.ndarray:
            dcResult[lsParamSets[i][0]] = (lsRMSE[-1], rmseR) # choose param for R as key
        else:
            dcResult[lsParamSets[i]] = (lsRMSE[-1], rmseR)
        
    print("%d trials have been finished" % nTrials)
    
    return dcResult

def invetigateLambda(strInPath, nMaxStep):
    
    #===========================================================================
    # initial params
    #===========================================================================
    arrAlphas = np.array([0.8,0.1,0.1]) 
    arrLambdas = np.array([0.4,0.3,0.3])
    f = 5
    dLearningRate = 0.01
    dTestRatio = 0.2
    
    #===========================================================================
    # set different settings
    #===========================================================================
    lsParamSets = [np.array([0.1, 0.3, 0.3]), \
                   np.array([0.3, 0.3, 0.3]), \
                   np.array([0.6, 0.3, 0.3]), \
                   np.array([0.9, 0.3, 0.3]), \
                   ]
    
    strParamName = 'lambda' 


    #===========================================================================
    # test different settings    
    #===========================================================================
    dcResult = multipleTrial(strInPath, lsParamSets, strParamName, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, dTestRatio)
    
    return dcResult

def testCMF():
    # setup
    arrAlphas = np.array([0.4, 0.3, 0.3])
    arrLambdas = np.array([0.2, 0.2, 0.2])
    f = 20
    dLearningRate = 0.01
    nMaxStep = 200
    
    
    # 10-fold cross validation
    dMinRmseR = 1.0
    lsBestTrainingRMSEs = None
    for i in xrange(1):
        
        # load data
        R = np.load('d:\\playground\\R_lite.npy')
        D = np.load('d:\\playground\\D_lite.npy')
        S = np.load('d:\\playground\\S_lite.npy')
        
        lsTrainingRMSE = []

        mu, bu, bv, U, P, V, Q, rmseR_test  = cmf.fit(R, D, S, arrAlphas, arrLambdas, f,\
                               dLearningRate, nMaxStep, lsTrainingRMSE, dTestRatio=0.2, bDebugInfo=True)
        
        if (rmseR_test < dMinRmseR):
            dMinRmseR = rmseR_test
            lsBestTrainingRMSEs = lsTrainingRMSE
    
    print('====testCMF finished====\n-->best training rmse:%f, test rmse:%f' % (lsTrainingRMSE[-1]['rmseR'], dMinRmseR) ) 
    cmf.visualizeRMSETrend(lsBestTrainingRMSEs)
        
if __name__ == '__main__':
    testCMF()





