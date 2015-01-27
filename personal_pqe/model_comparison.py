# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as prepro
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

def baseline(arrX, arrY, strModelName, dcModelParams, nFold=10, lsFeatureNames=None):
    '''
        Use given model as a baseline
        
        params:
                arrX - features
                arry - labels
                strModelName - model to usefdsclfds
                
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
        dcCurFold = {}
        # setup model
        model = None
        
        if (strModelName == 'GBRT'):
            if dcModelParams is not None:
                model = GradientBoostingRegressor(**dcModelParams)
            else:
                model = GradientBoostingRegressor()
            
        elif (strModelName == 'decision_tree_regression'):
            if dcModelParams is not None:
                model = DecisionTreeRegressor(**dcModelParams)
            else:
                model = DecisionTreeRegressor()
                
        elif (strModelName == 'random_forest_regression'):
            if dcModelParams is not None:
                model = RandomForestRegressor(**dcModelParams)
            else:
                model = RandomForestRegressor()
                
        elif (strModelName == 'linear_regression'):
            if dcModelParams is not None:
                model = LinearRegression(**dcModelParams)
            else:
                model = LinearRegression()
        else:
            print 'unsupported baseline!'
            break
        
        # fill nan
        arrX[np.isnan(arrX)] = 0.0
            
        # normalize features
        min_max_scaler = prepro.MinMaxScaler(copy=False)
        arrX = min_max_scaler.fit_transform(arrX)
        
        # split data
        arrX_train, arrX_test = arrX[arrTrainIndex], arrX[arrTestIndex]
        arrY_train, arrY_test = arrY[arrTrainIndex], arrY[arrTestIndex]
        
        # train
        model.fit(arrX_train, arrY_train)
        
        # test
        arrY_pred = model.predict(arrX_test)
        
        rmse = math.sqrt(mean_squared_error(arrY_test, arrY_pred))
        
        lsFeatureImportance = None
        if (lsFeatureNames is not None):
            assert(len(lsFeatureNames) == len(model.feature_importances_) )
            lsFeatureImportance = zip(lsFeatureNames, model.feature_importances_)
            
        dcCurFold['rmse'] = rmse
        dcCurFold['feature_importance'] = lsFeatureImportance
        
        print('-->%d fold cross validation: rmse=%f' % (i, rmse) )
        
        dcResults[i] = dcCurFold
        i = i+1
    
    return dcResults
    
if __name__ == '__main__':
    
    #===========================================================================
    # load data
    #===========================================================================
    strmtXPath = 'd:\\playground\\personal_qoe\\data\\sh\\mtX_0discre_rand1000.npy'
    strarrYPath = 'd:\\playground\\personal_qoe\\data\\sh\\arrY_0discre_rand1000.npy'
    
    strdfXPath = 'd:\\playground\\personal_qoe\\data\\sh\\dfX_0discre_rand1000'

    mtX = np.load(strmtXPath)
    arrY = np.load(strarrYPath)
    dfX = pd.read_pickle(strdfXPath)
    
    #===========================================================================
    # model setup
    #===========================================================================
#     strModelName = 'GBRT'
#     modelParams = {'n_estimators':100} 
    
#     strModelName = 'random_forest_regression'
#     modelParams = {'n_estimators':50} 
    
    strModelName = 'decision_tree_regression'
    modelParams = {'max_depth':4}

#     strModelName = 'linear_regression'
#     modelParams = {'normalize':False}

    #===========================================================================
    # test
    #===========================================================================
    nFold = 10
    dcResults = baseline(mtX, arrY, strModelName, modelParams, nFold, lsFeatureNames=dfX.columns.tolist())
    
    #===========================================================================
    # output
    #===========================================================================
    lsRMSEs = [i['rmse'] for i in dcResults.values()]
    dBestScore = np.min(lsRMSEs)
    dMeanScore = np.mean(lsRMSEs)
    dStd = np.std(lsRMSEs )
    print("cross validation is finished. \n--> best=%f, mean=%f, std=%f." % (dBestScore, dMeanScore, dStd ) )
    
    #===========================================================================
    # feature importance
    #===========================================================================
    print("*****Feature importance*****")
    for k,v in dcResults.iteritems():
        lsFeatureImportance = v['feature_importance']
        if (lsFeatureImportance is not None):
            srFeatureImportance = pd.Series([tp[1] for tp in lsFeatureImportance], \
                                            index=[tp[0] for tp in lsFeatureImportance])
            
            srFeatureImportance.sort(ascending=False)

            print("----fold%d: rmse=%f----" % (k, v['rmse']))
            print srFeatureImportance.iloc[:10]
