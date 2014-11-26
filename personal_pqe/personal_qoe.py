# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''

import numpy as np
import cmf.cmf_sgd as cmf_sgd
import sklearn.preprocessing as prepro

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

    bu, bv, U, P, V, Q  = cmf_sgd.fit(R, D, S, arrAlphas, arrLambdas, f, dLearningRate, nMaxStep, lsRMSE)
    
    cmf_sgd.visualizeRMSETrend(lsRMSE)
