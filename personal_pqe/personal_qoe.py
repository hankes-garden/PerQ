# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''

import numpy as np
import cmf.cmf_sgd as cmf
import pandas as pd

def multipleTrial(strRPath, strDPath, strSPath, \
				  lsParamSets, strParamName, \
                  arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, dTestRatio):
    #===========================================================================
    # load data & transform into matrix
    #===========================================================================
    mtR = np.load(strRPath)
    mtD = np.load(strDPath)
    mtS = np.load(strSPath)
    
    dcResults = {}
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
            raise RuntimeError
         
        #======================================================================
        # trial
        #======================================================================
        print("------------------------")
        print "arrAlphas", arrAlphas
        print 'arrlambdas', arrLambdas
        print 'f', f
        
        dcFoldResults, lsBestTrainingTrace  = cmf.cross_validate(mtR, mtD, mtS, arrAlphas, arrLambdas, f, nMaxStep, nFold=5)
        
    
        lsTestRMSE = [v['test'] for v in dcFoldResults.values() ]
        dMin = np.min(lsTestRMSE)
        dMax = np.max(lsTestRMSE)
        dMean = np.mean(lsTestRMSE)
        dStd = np.std(lsTestRMSE)
        print('-->min=%f, max=%f, mean=%f, std=%f' % (dMin, dMax, dMean, dStd))
        
        strKey = strParamName + " = " + str(lsParamSets)
        dcResults[strKey] = {'min':dMin, 'max':dMax, 'mean': dMean, 'std':dStd }
            
        
    print("%d trials have been finished" % nTrials)
    
    return dcResults

def findBestLambda(strRPath, strDPath, strSPath):
    #===========================================================================
    # initial params
    #===========================================================================
    arrAlphas = np.array([0.8,0.1,0.1]) 
    arrLambdas = np.array([0.4,0.3,0.3])
    f = 20
    dLearningRate = 0.0001
    dTestRatio = 0.3
    nMaxStep = 300
    
    #===========================================================================
    # set different settings
    #===========================================================================
    lsParamSets = [np.array([0.3, 0.3, 0.3]), \
                   np.array([0.6, 0.6, 0.6]), \
                   np.array([1.2, 1.2, 1.2]), \
                   np.array([2.4, 2.4, 2.4])
                   ]
    
    strParamName = 'lambda' 


    #===========================================================================
    # test different settings    
    #===========================================================================
    dcResult = multipleTrial(strRPath, strDPath, strSPath, \
                             lsParamSets, strParamName, \
                             arrAlphas, arrLambdas, \
                             f, dLearningRate, nMaxStep, dTestRatio)
    
    return dcResult

def findBestAlpha(strRPath, strDPath, strSPath):
    #===========================================================================
    # initial params
    #===========================================================================
    arrAlphas = np.array([0.8,0.1,0.1]) 
    arrLambdas = np.array([2,2,2])
    f = 20
    dLearningRate = 0.0001
    dTestRatio = 0.3
    nMaxStep = 400
    
    #===========================================================================
    # set different settings
    #===========================================================================
    lsParamSets = [np.array([0.33, 0.33, 0.33]), \
                   np.array([0.5, 0.25, 0.25]), \
                   np.array([0.6, 0.2, 0.2]), \
                   np.array([0.8, 0.1, 0.1]), \
                   np.array([1, 1, 1]), 
                   ]
    
    strParamName = 'alpha' 


    #===========================================================================
    # test different settings    
    #===========================================================================
    dcResult = multipleTrial(strRPath, strDPath, strSPath, lsParamSets, strParamName, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, dTestRatio)
    
    return dcResult
        
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	dc = findBestAlpha('d:\\playground\\sh_xdr\\R_top500.npy', \
                  'd:\\playground\\sh_xdr\\D_top500.npy', 
                  "d:\\playground\\sh_xdr\\S_top500.npy")
	df = pd.DataFrame(dc)
	df.T.plot()
	plt.show()