# -*- coding: utf-8 -*-
'''
Brief Description: 

@author: jason
'''

import common_function 

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
import operator
import matplotlib.pyplot as plt


g_lsSelectedColumns = [0,4,7,8,9,10,16,18,20,21]
g_lsNumericColumns = [0,16,18,20,21]
g_lsCategoricalColumns = []

g_modelParams = {'n_estimators':100, 'loss':'lad'} 
 
for i in g_lsSelectedColumns:
    if i not in g_lsNumericColumns:
        g_lsCategoricalColumns.append(i)
        

def recordTruePredict(wfn, gbr, x_valid, y_valid):
    wf = open(wfn, 'w') 
    y_predict = [] 
    relatedError = 0.0 
    for(features, value) in zip(x_valid, y_valid):
        p_val = gbr.predict(features) 
        y_predict.append(float(p_val)) 
        oline = str(value) + ',%.4f\n' % p_val 
        relatedError += abs(value - p_val) / value 
        wf.write('Relative Absolute Error: %.4f' % (relatedError/len(y_valid))) 
    wf.close() 
    print 'Relative Absolute Error: %.4f' % (relatedError/len(y_valid))
    
    
def preprocessDataSet(dfData, lsSelectedColumns, lsNumericColumns, lsCategoricalColumns):
    '''
        This function selects given columns, maps categorical and integer columns.
    '''
    
    # fill nan first
    dfData.fillna(0, inplace=True)
    
    # numeric columns
    dfNumVariables = dfData.iloc[:, lsNumericColumns]
    arrNumVariables = dfNumVariables.as_matrix()
    
    # categorical columns
    dfCateVariables = dfData.iloc[:,lsCategoricalColumns]
    vec =  DictVectorizer(sparse=False)
    arrCateFeatures = vec.fit_transform(dfCateVariables.T.to_dict().values())
    
    # setup training set
    arrX = np.concatenate((arrNumVariables, arrCateFeatures), axis=1)
    lsVariableNames = dfNumVariables.columns.tolist()
    lsVariableNames += vec.get_feature_names()
    
    arrY = dfData['DOWNLOAD_RATIO'].values
    
    return arrX, arrY, lsVariableNames

def trainModel(arrX, arrY, params):
    # setup regressor
    gbr = GradientBoostingRegressor(**params) 
    
    # fit
    print('start to train model ...') 
    gbr.fit(arrX, arrY) 
    print('finish training model.')
    
    return gbr

def crossValidate(arrX, arrY, params, nFold):
    # 10-fold cross validation
    dcModels = {}
    kf = cross_validation.KFold(len(arrY), n_folds=10)
    for arrTrainIndex, arrTestIndex in kf:
        # setup regressor
        model = GradientBoostingRegressor(**params)
        
        arrXTrain, arrXTest = arrX[arrTrainIndex], arrX[arrTestIndex]
        arrYTrain, arrYTest = arrY[arrTrainIndex], arrY[arrTestIndex]
        
        # train
        model.fit(arrXTrain, arrYTrain)
        
        # test
        fScore = model.score(arrXTest, arrYTest)
        
        
        dcModels[model] = fScore
            
    return dcModels

def getVariableImportance(model, lsTrainingFeatureNames):
    '''
        This function find the importance of unmapped variables
    '''
    dcVariableImportance = {}
    lsFeatureImportance = model.feature_importances_.tolist()
    
    assert(len(lsFeatureImportance) == len(lsTrainingFeatureNames) )
    
    nLen = len(lsFeatureImportance)
    for i in range(0, nLen):
        strKey = lsTrainingFeatureNames[i].split('=')[0]
        common_function.updateDictBySum(dcVariableImportance, strKey, lsFeatureImportance[i])
    
    return dcVariableImportance

def predict(model, arrX, arrY):
    # test
    mae_test = mean_absolute_error(arrY, model.predict(arrX)) 
    mse_test = mean_squared_error(arrY, model.predict(arrX)) 
    print("MAE: %.2f, MSE:%.2f. " % (mae_test, mse_test) )
    
def drawVariableImportance(dfVariableImportance):
    fig, ax=plt.subplots()
    dfVariableImportance.T.iloc[:,range(0, (len(dfVariableImportance.T.columns) -1 ) )].plot(ax=ax)
    dfVariableImportance.T.iloc[:,len(dfVariableImportance.T.columns)-1].plot(ax=ax, style='r--', linewidth=4.0)
    ax.set_xticklabels(dfVariableImportance.T.index.tolist(), rotation=90)
    plt.show()
    
def execute(strInPath, strOutPath, bSerialize=False):
    '''
        this function trains model for each user and compares with the model trained from all users
    '''
    # find xdr
    lsXDR = common_function.getFileList(strInPath, "out")
    
    dcVariableImportance = {}           # variable importance of each personal model
    dcModels = {}                       # dict of personal models
    for xdr in lsXDR:
        # load data
        print("processing %s..." % xdr)
        dfData = pd.read_csv(xdr, sep='|', \
                             names= ['BEGIN_TIME','BEGIN_TIME_MSEL','MSISDN','IMSI','SERVER_IP',\
                                     'SERVER_PORT','APN','PROT_CATEGORY','PROT_TYPE','LAC','SAC',\
                                     'CI','IMEI','RAT','HOST','STREAMING_URL','STREAMING_FILESIZE',\
                                     'STREAMING_DW_PACKETS','STREAMING_DOWNLOAD_DELAY','ASSOCIATED_ID',\
                                     'L4_UL_THROUGHPUT','L4_DW_THROUGHPUT', 'use_less'] )
        del dfData['use_less']
        dfData['DOWNLOAD_RATIO'] = dfData.iloc[:,17]*1.0/dfData.iloc[:,16]
        
        strIMSI = xdr.split('/')[-1].split('.')[0]
        
        # prepare data set
        arrX, arrY, lsTrainingFeatureNames = preprocessDataSet(dfData, g_lsSelectedColumns, \
                                                                                g_lsNumericColumns, \
                                                                                g_lsCategoricalColumns)
        
#         # train model
#         model = trainModel(arrX, arrY, g_modelParams)
#         dcVariableImportance[strIMSI] = getVariableImportance(model, lsTrainingFeatureNames)
#          
#         # test
#         mse = mean_squared_error(arrY, model.predict(arrX) )
#         mae = mean_absolute_error(arrY, model.predict(arrX) )
#         print("MSE: %.4f, MAE: %.4f" % (mse, mae) )
        
        # cross validation
        dcPersonalModels = crossValidate(arrX, arrY, g_modelParams, 10)
        bestModel, fBestScore = max(dcPersonalModels.iteritems(), key=operator.itemgetter(1) )
        dcVariableImportance[strIMSI] = getVariableImportance(bestModel, lsTrainingFeatureNames)
        
        dcModels[strIMSI] = (fBestScore, bestModel)

        print("model:%s, #record=%d, best=%0.2f, mean=%.2f, std=%0.2f. \n)" % \
              (strIMSI, len(arrY), fBestScore, np.mean(dcPersonalModels.values() ), np.std(dcPersonalModels.values() ) ) )
    
    dfVariableImportance = pd.DataFrame(dcVariableImportance).T

    # serialize models
    if(bSerialize is True):
        common_function.serialize2File(strOutPath+'serDcModels.out', dcPersonalModels)
        dfVariableImportance.to_csv(strOutPath+'dfVariableImportance_all.out')
        
    return dcModels, dfVariableImportance
        
    
    
    
    
    
    