# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''

import numpy as np
import cmf.cmf_sgd as cmf_sgd

def test(strDataPath):
    #===========================================================================
    # load data
    #===========================================================================
    
    #===========================================================================
    # prepare matrix
    #===========================================================================
    if (strDataPath.endswith("/") is not True):
        strDataPath.append("/")
    D = np.load(strDataPath+"D.npy")
    S = np.load(strDataPath+"S.npy")
    R = np.load(strDataPath+"R.npy")
    
    #===========================================================================
    # cmf
    #===========================================================================
    arrAlphas = np.array([0.7, 0.0, 0.3])
    arrLambdas = np.array([0.4, 0.3, 0.3])
    f = 5
    dLearningRate = 0.05
    nMaxStep = 100
    lsRMSE=[]

    bu, bv, U, P, V, Q  = cmf_sgd.cmf(D, S, R, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsRMSE)
    
    cmf_sgd.visualizeRMSETrend(lsRMSE)

if __name__ == '__main__':
    pass