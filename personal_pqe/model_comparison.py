# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''
import numpy as np
import operator

from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as prepro
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
import math

def transformMatrices2FlatTable(strRPath, strDPath, strSPath):
    mtR = np.load(strRPath)
    mtD = np.load(strDPath)
    mtS = np.load(strSPath)
    
    
    mtFeatures = None
    lsLabels = []
    
    R = np.where(mtR==255, 0.0, mtR*1.0/100.0)
    arrRows, arrCols = np.nonzero(R)
    
    nLen = len(arrRows)
    for i in xrange(nLen):
        nRowID = arrRows[i]
        nColID = arrCols[i]
        
        lsLabels.append( R[nRowID, nColID] )
        arrFeatures = np.hstack([mtD[nRowID,:], mtS[nColID, :] ])
        if (mtFeatures is None):
            mtFeatures = arrFeatures
        else:
            mtFeatures = np.vstack([mtFeatures, arrFeatures] )
    
    return mtFeatures, np.array(lsLabels)

def baseline(arrX, arrY, strModelName, dcModelParams, nFold=10):
    '''
        Use given model as a baseline
        
        params:
                arrX - features
                arry - labels
                strModelName - model to use
                dcModelParams - model params
                nFold - # fold
        return:
                dcResults - a dict of rmse of each fold
    '''
    
    #===========================================================================
    # cross validation
    #===========================================================================
    dcResults = {}
    kf = cross_validation.KFold(len(arrY), nFold)
    
    print('start to cross validate...')
    i = 0
    for arrTrainIndex, arrTestIndex in kf:
        # setup model
        model = None
        if (strModelName == 'GBRT'):
            
            # fill nan
            arrX[np.isnan(arrX)] = 0.0
            
            # normalize features
            min_max_scaler = prepro.MinMaxScaler(copy=False)
            arrX = min_max_scaler.fit_transform(arrX)
        
            if dcModelParams is not None:
                model = GradientBoostingRegressor(**dcModelParams)
            else:
                model = GradientBoostingRegressor()
            
        elif (strModelName == 'decision_tree_regression'):
            # fill nan
            arrX[np.isnan(arrX)] = 0.0
            
            # normalize features
            min_max_scaler = prepro.MinMaxScaler(copy=False)
            arrX = min_max_scaler.fit_transform(arrX)
            
            if dcModelParams is not None:
                model = DecisionTreeRegressor(**dcModelParams)
            else:
                model = DecisionTreeRegressor()
        else:
            print 'unsupported baseline!'
            break
        
        # split data
        arrX_train, arrX_test = arrX[arrTrainIndex], arrX[arrTestIndex]
        arrY_train, arrY_test = arrY[arrTrainIndex], arrY[arrTestIndex]
        
        # train
        model.fit(arrX_train, arrY_train)
        
        # test
        arrY_pred = model.predict(arrX_test)
        
        rmse = math.sqrt(mean_squared_error(arrY_test, arrY_pred))
        
        print('-->%d fold cross validation: rmse=%f' % (i, rmse) )
        dcResults[i] = rmse
        
        i = i+1
    
    return dcResults
    
if __name__ == '__main__':
    
    #===========================================================================
    # load data
    #===========================================================================
    strRPath = 'd:\\playground\\personal_qoe\\sh\\R_no_discretize_top100.npy'
    strDPath = 'd:\\playground\\personal_qoe\\sh\\D_no_discretize_top100.npy'
    strSPath = 'd:\\playground\\personal_qoe\\sh\\S_no_discretize_top100.npy'
    
    strFlattenTable = 'd:\\playground\\personal_qoe\\sh\\mtX_0discre_top100.npy'
    strFlattenLabel = 'd:\\playground\\personal_qoe\\sh\\arrY_0discre_top100.npy'

    mtX = np.load(strFlattenTable)
    arrY = np.load(strFlattenLabel)
    
    #===========================================================================
    # model setup
    #===========================================================================
#     strModelName = 'GBRT'
#     modelParams = {'n_estimators':100} 
    
    strModelName = 'decision_tree_regression'
    modelParams = {'max_depth':4}
    
    #===========================================================================
    # test
    #===========================================================================
    dcResults = baseline(mtX, arrY, strModelName, modelParams, 5)
    
    #===========================================================================
    # output
    #===========================================================================
    fBestScore = min(dcResults.values() )
    
    print("cross validation is finished. \n--> best=%f, mean=%f, std=%f." % \
          (fBestScore, np.mean(dcResults.values() ), np.std(dcResults.values() ) ) )

