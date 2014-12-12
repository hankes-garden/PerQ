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
    
def multipleTrial(strInPath, lsParamSets, strParamName, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep=500):
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
        bu, bv, U, P, V, Q  = cmf.fit(R, D, S, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsRMSE )
        
        if type(lsParamSets[i]) is np.ndarray:
            dcResult[lsParamSets[i][0]] = lsRMSE[-1] # choose param for R as key
        else:
            dcResult[lsParamSets[i]] = lsRMSE[-1]
        
    print("%d trials have been finished" % nTrials)
    
    return dcResult

def invetigateLambda(strInPath):
    
    #===========================================================================
    # initial params
    #===========================================================================
    arrAlphas = np.array([0.8,0.1,0.1]) 
    arrLambdas = np.array([0.4,0.3,0.3])
    f = 5
    dLearningRate = 0.01
    nMaxStep = 400
    
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
    dcResult = multipleTrial(strInPath, lsParamSets, strParamName, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep)






