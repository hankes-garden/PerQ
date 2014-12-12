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

g_strModuleNameForAllUser = 'all users'

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
    
    
def preprocessDataSet(dfData, lsNumericColumns, lsCategoricalColumns, strLabelColumn):
    '''
        This function selects given columns, maps categorical and integer columns.
    '''
    
    # fill nan first
    dfData.fillna(0, inplace=True)
    
    # numeric columns
    dfNumVariables = dfData.ix[:, lsNumericColumns]
    arrNumVariables = dfNumVariables.as_matrix()
    
    # categorical columns
    dfCateVariables = dfData.ix[:,lsCategoricalColumns]
    vec =  DictVectorizer(sparse=False)
    arrCateFeatures = vec.fit_transform(dfCateVariables.T.to_dict().values())
    
    # setup training set
    arrX = np.concatenate((arrNumVariables, arrCateFeatures), axis=1)
    lsVariableNames = dfNumVariables.columns.tolist()
    lsVariableNames += vec.get_feature_names()
    
    arrY = dfData[strLabelColumn].values
    
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
    
def validateOnSH(strInPath, strOutPath, bSerialize=False):
    '''
        this function validates user diversity on Shanghai data set
        
        param:
                strInPath  - path for separate files of top users
                strOutPath - path to serialize model
        
        Note: we need another script to distributes records of top users into separate files,
              and this function will only read separate files from strInPath
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
                                                                                g_lsCategoricalColumns,\
                                                                                'DOWNLOAD_RATIO')
        
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

def validateOnNJ(strInPath, strOutPath, bSerialize=False):
    '''
        this function validate user diversity on Nanjin data set
        return:
                dcModels - {uid: model}
                dfVariableImportance.T - row:uid, column:variable importance
    '''
    #===========================================================================
    # setup transform rules
    #===========================================================================
    lsColumns2Delete = ['up', 'down', 'playtimes', 'normalized popratio', 'month', 'day',\
                        'starttime', 'endtime', 'signal strength', 'data variance1', 'data variance2',\
                        'data variance3', 'data variance4', 'average first 10s data rate',\
                        'absolute deviation of data variance (Median)',\
                        'absolute deviation of data variance (Mode)']
    
    dcColumns2Discretize = {'rebuffer time': [10, 20, 30, 40, 50], \
                            'totaltime ': [10*60, 30*60, 60*60],\
                            'average data rate': [20,40,60,80,100,120],\
                            'last 10s rateratio':[0.9, 1.5],\
                            'popratio': [0.5, 0.8],\
                            'startup delay': [10, 30, 60, 90, 120],\
                            'data variance':[10,20,30,40],\
                            'average signal strength': [-20, -40, -60],\
                            'first 10s rateratio': [0.5, 1.0, 1.5, 2.0],\
                            'absolute deviation of data variance (Mean)':[10,20,30],\
                            'p-norm of data variance(P=3)': [50,100,150,200,250],\
                            }
    
    lsColumns2Vectorize = ['AP MAC', 'week day', 'PC/mobile', 'video website', 'gender', 'grade']
    strLabelColumnName = "viewing time ratio"
    strUserIDColumnName = "user MAC"
    lsUserProfileColumns = ['gender', 'grade']
    strVideoIDColumnName = "vid"
    lsVideoQualityColumns = ['rebuffer time', 'totaltime ', 'average data rate', 'last 10s rateratio',\
                             'popratio', 'startup delay', 'AP MAC', 'week day', 'data variance',\
                             'PC/mobile', 'resolution', 'average signal strength',  'first 10s rateratio',\
                             'absolute deviation of data variance (Mean)', 'p-norm of data variance(P=3)',\
                             'video website']
    
    #===========================================================================
    # load & filter invalid rows and columns
    #===========================================================================
    dfAllData = pd.read_csv(strInPath)
    
    # filter invalid rows out
    dfAllData = dfAllData[ (dfAllData['viewing time ratio']>=0.0) & (dfAllData['viewing time ratio']<=1.0) & \
                    (dfAllData['user MAC'] != np.nan) & (dfAllData['vid'] != np.nan) ]
    
    for col in lsColumns2Delete:
        del dfAllData[col]
        
    print ("%d rows * %d columns have been loaded." % (len(dfAllData), len(dfAllData.columns) ) )
    
    #===========================================================================
    # find Top Users
    #===========================================================================
    lsDataofEachUser = []
    arrUniqueUsers = dfAllData[strUserIDColumnName].unique()
    for uid in arrUniqueUsers:
        _df = dfAllData[dfAllData[strUserIDColumnName] == uid]
        lsDataofEachUser.append( (len(_df), _df) )
    
    lsDataofEachUser.sort(key=lambda x:x[0], reverse=True)
    lsTopUsers = lsDataofEachUser[:5]
    
    #===========================================================================
    # train personal model for top 10 users
    #===========================================================================
    dcVariableImportance = {}           # variable importance of each personal model
    dcModels = {}                       # dict of personal models
    uNum = 0
    for (nDataNum, dfUserRecord) in lsTopUsers:
        
        strUid = dfUserRecord[strUserIDColumnName].iloc[0]
        print("processing %s (%d rows)..." % (strUid, nDataNum) )
        
        del dfUserRecord[strUserIDColumnName]
        del dfUserRecord[strVideoIDColumnName]
                
        # prepare data set
        arrX, arrY, lsTrainingFeatureNames = preprocessDataSet(dfUserRecord, dcColumns2Discretize.keys(), \
                                                               lsColumns2Vectorize, strLabelColumnName)
        
        # train model
        model = trainModel(arrX, arrY, g_modelParams)
        dcVariableImportance["user %d" % uNum] = getVariableImportance(model, lsTrainingFeatureNames)
           
        # test
        mse = mean_squared_error(arrY, model.predict(arrX) )
        mae = mean_absolute_error(arrY, model.predict(arrX) )
        print("    MSE: %.4f, MAE: %.4f" % (mse, mae) )
        uNum += 1
        
#         # cross validation
#         dcPersonalModels = crossValidate(arrX, arrY, g_modelParams, 3)
#         bestModel, fBestScore = max(dcPersonalModels.iteritems(), key=operator.itemgetter(1) )
#         dcVariableImportance[strUid] = getVariableImportance(bestModel, lsTrainingFeatureNames)
#         
#         dcModels[strUid] = (fBestScore, bestModel)
# 
#         print("model:%s, #record=%d, best=%0.2f, mean=%.2f, std=%0.2f. \n)" % \
#               (strUid, len(arrY), fBestScore, np.mean(dcPersonalModels.values() ), np.std(dcPersonalModels.values() ) ) )

    #===========================================================================
    # train model  for all users    
    #===========================================================================
    print("processing all data (%d rows)..." % (len(dfAllData)) )
    
    del dfAllData[strUserIDColumnName]
    del dfAllData[strVideoIDColumnName]
            
    # prepare data set
    arrX, arrY, lsTrainingFeatureNames = preprocessDataSet(dfAllData, dcColumns2Discretize.keys(), \
                                                           lsColumns2Vectorize, strLabelColumnName)
    
    # train model
    model = trainModel(arrX, arrY, g_modelParams)
    dcVariableImportance[g_strModuleNameForAllUser] = getVariableImportance(model, lsTrainingFeatureNames)
       
    # test
    mse = mean_squared_error(arrY, model.predict(arrX) )
    mae = mean_absolute_error(arrY, model.predict(arrX) )
    print("    MSE: %.4f, MAE: %.4f" % (mse, mae) )
    
    dfVariableImportance = pd.DataFrame(dcVariableImportance).T


#     # serialize models
#     if(bSerialize is True):
#         common_function.serialize2File(strOutPath+'serDcModels.out', dcPersonalModels)
#         dfVariableImportance.to_csv(strOutPath+'dfVariableImportance_all.out')
        
    return dcModels, dfVariableImportance.T


def visualizeVariableImportance(dfVariableImportance):
    
    # create single plot in a figure
    ax0 = plt.figure().add_subplot(111) # note it is figure(), not Figure()!
    axDraw = dfVariableImportance.plot(ax=ax0, ls='--')
    
    # set line for all user
    lines = axDraw.get_lines()
    for line in lines:
        if line.get_label() == 'all viewers':
            line.set_linewidth(2)
            line.set_linestyle('-')
    
    ax0.set_ylabel('weight')
    plt.xticks(range(len(dfVariableImportance.index)),dfVariableImportance.index.tolist())
    plt.setp(plt.xticks()[1], rotation=90)
    plt.legend(loc='upper left')
    plt.show()
        
if __name__ == '__main__':
    dcModels, dfVariableImportance = validateOnNJ("d:\\playground\\sh_xdr\\nj\\all_with_router_info.csv", "d:\\playground\\sh_xdr\\nj\\observation_validation\\", False)
    visualizeVariableImportance(dfVariableImportance)
    
    
    
    
    