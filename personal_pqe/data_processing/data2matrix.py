# -*- coding: utf-8 -*-
'''
Brief Description: 
        This module transform XDR data records into sparse matrix
@author: jason
'''

import pandas as pd
import numpy as np
import sklearn.preprocessing as prepro

g_dcColumns2Discritize = {'BEGIN_TIME': [7*3600,9*3600,12*3600,14*3600,18*3600,20*3600], \
                          'STREAMING_FILESIZE': [10.0, 50.0, 100.0, 200.0, 300.0, 400.0],\
                          'STREAMING_DW_SPEED': [50.0, 100,0, 150.0, 200.0, 250.0, 300.0],\
                          'TCP_RTT': [500, 1000, 1500, 2000, 2500, 3000]
                          }

g_lsColumns2Digitalize = ['APN', 'PROT_TYPE', 'LOCATION', 'RAT', 'HOST']
g_dcDigitMappingTable = {}

def discretizeColumnEx(srColumn, lsBins, func=None):
    '''
        discretize value based on predefined bins
    '''
    strName = srColumn.name
    if lsBins is None:
        return None
    else:
        if func is not None:
            srColumn = srColumn.apply(func)
        
        lsBins.append(srColumn.min() )
        lsBins.append(srColumn.max() )
        lsBins.sort()
        
        arrCuts = pd.cut(srColumn, bins=lsBins, labels=False)
    
    return arrCuts

def discretizeColumn(srColumn, func=None):
    '''
        discretize value based on predefined bins
    '''
    strName = srColumn.name
    lsBins = g_dcColumns2Discritize.get(strName, None)
    if lsBins is None:
        return None
    else:
        if func is not None:
            srColumn = srColumn.apply(func)
        
        lsBins.append(srColumn.min() )
        lsBins.append(srColumn.max() )
        lsBins.sort()
        
        arrCuts = pd.cut(srColumn, bins=lsBins, labels=False)
    
    return arrCuts

def digitalizeColumnEx(srColumn):
    '''
        digitalize string columns into integer   
    '''
        
    # digitalise
    arrUniqueValues, arrIndex = np.unique(srColumn, return_inverse=True)
    
    dcMappingtable = {}
    for i in xrange(len(arrUniqueValues)):
        dcMappingtable[i] = arrUniqueValues[i]
    
    return arrIndex, dcMappingtable

def digitalizeColumn(srColumn):
    '''
        digitalize string columns into integer   
    '''
    # no need to digitalise
    if ( not (srColumn.name in g_lsColumns2Digitalize) or not (isinstance(srColumn[0], basestring)) ):
        return (None,None)
        
    # digitalise
    arrUniqueValues, arrIndex = np.unique(srColumn, return_inverse=True)
    
    dcMappingtable = {}
    for i in xrange(len(arrUniqueValues)):
        dcMappingtable[i] = arrUniqueValues[i]
    
    return arrIndex, dcMappingtable

def getVideoString(srRow):
    return srRow.drop(['IMSI', 'RATIO'], inplace=False).to_string()

def transform2VideoQualityMatrixEx(dfData, lsColumns2Delete, dcColumns2Discretize, lsColumns2Vectorize, \
                                   strLabelColumnName, \
                                   strUserIDColumnName, lsUserProfileColumns, \
                                   strVideoIDColumnName, lsVideoQualityColumns):
    '''
        This function transform xdr data into user, video, rating matrices.
        - delete useless columns
        - discretize continuous data
        - digitalize & mapping categorical data
        - transfrom into R, dfD, dfS matrices
        
    '''
    # TODO: how to handle NAN?
    
    
    #===========================================================================
    # delete useless columns
    #===========================================================================
    print("start to delete useless columns...")
    for strColName in lsColumns2Delete:
        del dfData[strColName]
    
    #===========================================================================
    # discretize continuous data
    #===========================================================================
    print("start to discretize continuous data...")
    for (strColName,lsBins) in dcColumns2Discretize.iteritems():
        sCol = dfData[strColName]

        arrCuts = discretizeColumnEx(sCol, lsBins, None)
        if arrCuts is not None:
            dfData[strColName] = arrCuts
        else:
            print ("discretize")
        
    
    #===========================================================================
    # mapping categorical data
    #===========================================================================
    print("start to mapping categorical data...")
    for strColName in lsColumns2Vectorize:
        sCol = dfData[strColName]
        
        # digitalize categorical data
        arrDigits, dcMappingTable = digitalizeColumnEx(sCol)
        g_dcDigitMappingTable[sCol.name] = dcMappingTable
        
        # vectorize
        enc = prepro.OneHotEncoder()
        mtEncoded = enc.fit_transform(pd.DataFrame(arrDigits) ).toarray()
        
        # add new columns & update corresponding list
        lsColumnName2Update = None
        if strColName in lsUserProfileColumns:
            lsColumnName2Update = lsUserProfileColumns
        elif (strColName in lsVideoQualityColumns):
            lsColumnName2Update = lsVideoQualityColumns
        else:
            print "useless column, why do you map it ?"
            
        lsColumnName2Update.remove(strColName)
        for ci in xrange(mtEncoded.shape[1]):
            strName = "%s_%d" % (strColName, ci)
            dfData[strName] = mtEncoded[:,ci]
            lsColumnName2Update.append(strName)
        
        # delete original column
        del dfData[strColName]
        
        # TODO: find way to mapping it back!
    
    #===========================================================================
    # transfrom into D
    #===========================================================================
    print("start to transfrom into D...")
#     arrUID = np.array(dfData[strUserIDColumnName].tolist() )
#     arrUniqueUsers, arrIndex = np.unique(arrUID, return_index=True)
#     dfD = dfData.loc[arrIndex.astype(int).tolist()][ lsUserProfileColumns+[strUserIDColumnName,] ]
    
    # delete duplicated rows, note: don't change the index
    dfD = dfData.drop_duplicates(strUserIDColumnName)[ lsUserProfileColumns+[strUserIDColumnName,] ]
    
    lsUsers = dfD[strUserIDColumnName].tolist()
    
    del dfD[strUserIDColumnName] # do not include uid in D matrix
    
    #===========================================================================
    # transform into S
    #===========================================================================
    print("start to transform into S...")
    dcS ={}
    dcVideoRatio = {} # to remember labels attached to each video 
    nCount = 0
    for ind, row in dfData.iterrows():
        if (nCount % 100 == 0):
            print("%.2f%%" % (nCount*100.0/len(dfData)) )
            
        sVideo = row.loc[ lsVideoQualityColumns+[strVideoIDColumnName, ] ]
        strKey = sVideo.to_string()         # use vid + video qualities as key
        uid = row[strUserIDColumnName]      # user of current row
        dLabel = row[strLabelColumnName]    # label of current row
        
        # update dfS
        dcS[strKey] = sVideo
        
        # prepare for R
        dcUserLabel = dcVideoRatio.get(strKey)
        if dcUserLabel is None: # if already exist, then add
            dcUserLabel = {}
            dcVideoRatio[strKey] = dcUserLabel
        
        dcUserLabel[uid] = dLabel
        nCount += 1
    
    dfS = pd.DataFrame(dcS).T.convert_objects(convert_numeric = True)
    lsVideos = dfS.index.tolist()
    
    del dfS[strVideoIDColumnName] # do not include vid in S
   
    
    #===========================================================================
    # transform into R
    #===========================================================================
    print("start to transform into R...")
    dfR = pd.DataFrame(index=lsUsers)
    for strVideoKey,dcUserRatio in dcVideoRatio.iteritems():
        sRatios = pd.Series(index=lsUsers)
        for uid, dLabel in dcUserRatio.iteritems():
            sRatios[uid] = dLabel
        
        dfR[strVideoKey] = sRatios

    #===========================================================================
    # sort to ensure R, D, S are in the same order
    #===========================================================================
    dfR = dfR[lsVideos]
    
    print("transformation is finished!")
    
    return dfR.as_matrix(), dfD.as_matrix(), dfS.as_matrix()


def transform2VideoQualityMatrix(df):
    '''
        This function transform xdr data into video-matrix quality
        1. discretize continuous data
        2. vectorize categorical data
        3. transfrom into R, dfS matrices
         
    '''
   
    #===========================================================================
    # intial columns:
    #          ['BEGIN_TIME','BEGIN_TIME_MSEL','MSISDN','IMSI',\
    #          'SERVER_IP','SERVER_PORT','APN','PROT_CATEGORY',\
    #          'PROT_TYPE','LAC','SAC','CI','IMEI','RAT','HOST',\
    #          'STREAMING_URL','STREAMING_FILESIZE','STREAMING_DW_PACKETS',\
    #          'STREAMING_DOWNLOAD_DELAY','ASSOCIATED_ID','L4_UL_THROUGHPUT',\
    #          'L4_DW_THROUGHPUT', 'INTBUFFER_FULL_FLAG', 'TCP_RTT', \
    #          'GET_STREAMING_DELAY', 'INTBUFFER_FULL_DELAY', 'SID', 'use_less'] )
    #                                   
    #===========================================================================
    
    #===========================================================================
    # copy data & handle invalid values
    #===========================================================================
    dfStreaming = df.copy(deep=True)
    
    try:
        del dfStreaming['BEGIN_TIME_MSEL']
        del dfStreaming['MSISDN']
        del dfStreaming['SERVER_IP']
        del dfStreaming['SERVER_PORT']
        del dfStreaming['PROT_CATEGORY']
        del dfStreaming['IMEI']
        del dfStreaming['STREAMING_URL']
        del dfStreaming['ASSOCIATED_ID']
        del dfStreaming['SID']
        
        # TODO: new data set may has these two features
        del dfStreaming['L4_UL_THROUGHPUT']
        del dfStreaming['L4_DW_THROUGHPUT']
        del dfStreaming['INTBUFFER_FULL_FLAG']
        del dfStreaming['GET_STREAMING_DELAY']
        del dfStreaming['INTBUFFER_FULL_DELAY']
        
    except Exception as err:
        print err
    
    #===========================================================================
    # add columns
    #===========================================================================
    # RATIO
    dfStreaming['RATIO'] = dfStreaming['STREAMING_DW_PACKETS']*1.0/dfStreaming['STREAMING_FILESIZE']
    
    # LOCATION
    dfStreaming['LOCATION'] = dfStreaming['LAC'].apply(str) + '-' \
                                + dfStreaming['SAC'].apply(str)\
                                + '-' + dfStreaming['CI'].apply(str)
    del dfStreaming['LAC']
    del dfStreaming['SAC']
    del dfStreaming['CI']
    
    # DW_SPEED
    dfStreaming['STREAMING_DOWNLOAD_DELAY'].replace(0, 1, inplace=True)
    dfStreaming['STREAMING_DW_SPEED'] = dfStreaming['STREAMING_DW_PACKETS']*1.0/dfStreaming['STREAMING_DOWNLOAD_DELAY']
    del dfStreaming['STREAMING_DW_PACKETS']
    del dfStreaming['STREAMING_DOWNLOAD_DELAY']
    
    #===========================================================================
    # transform data
    #===========================================================================
    for col in dfStreaming.columns:
        srColumn = dfStreaming[col]
        
        #discretize continuous data
        func = None
#         if srColumn.name == 'BEGIN_TIME':
#             func = cf.getSecondofDay
            
        arrCuts = discretizeColumn(srColumn, func)
        if arrCuts is not None:
            dfStreaming[col] = arrCuts
        
        # digitalize categorical data
        arrDigits, dcMappingTable = digitalizeColumn(srColumn)
        if arrDigits is not None:
            dfStreaming[col] = arrDigits
            g_dcDigitMappingTable[srColumn.name] = dcMappingTable
            
        # TODO: no need to vectorize categorical data?
            
            
            
     #==========================================================================
     # data now 
     #['BEGIN_TIME',
     # 'IMSI',
     # 'APN',
     # 'PROT_TYPE',
     # 'RAT',
     # 'HOST',
     # 'STREAMING_FILESIZE',
     # 'TCP_RTT',
     # 'RATIO',
     # 'LOCATION',
     # 'STREAMING_DW_SPEED']
     #==========================================================================

    #===========================================================================
    # construct matrices dfS, R
    #===========================================================================
    
    # find duplicated rows
    lsVideoFeatures = dfStreaming.columns.tolist()
    lsVideoFeatures.remove('IMSI')
    lsVideoFeatures.remove('RATIO')
#     arrDuplicatedIndex = dfStreaming.duplicated(lsVideoFeatures)
#     
#     
#     nUsers = len(arrDistinctUsers)
#     nVideos = len(arrDuplicatedIndex) - arrDuplicatedIndex.sum()
    
    dcDistinctVideos = {}
    nTotalLen = len(dfStreaming)
    for index, row in dfStreaming.iterrows():
        if (index%100==0):
            print "1st", index*1.0/nTotalLen
        strKey = getVideoString(row)
        lsCurrentVideoWatchingRecords = dcDistinctVideos.get(strKey, None)
        if (lsCurrentVideoWatchingRecords is None):
            lsCurrentVideoWatchingRecords = []
        
        lsCurrentVideoWatchingRecords.append(index)
        dcDistinctVideos[strKey] = lsCurrentVideoWatchingRecords
        
    # R
    arrDistinctUsers = dfStreaming['IMSI'].unique()
    arrDistinctVideos = dcDistinctVideos.keys()
    dfR = pd.DataFrame()
#     dfR.fillna(0.0, inplace=True)
#     for index, row in dfStreaming.iterrows():
#         if (index%100==0):
#             print "2nd", index*1.0/nTotalLen
#         strKey = getVideoString(row)
#         dRatio = row['RATIO']
#         imsi = row['IMSI']
#         dfR.loc[imsi,strKey] = dRatio
    vi = 0
    nTotalCols = len(arrDistinctVideos)
    for v in arrDistinctVideos:
        
        vi += 1
        print '2nd: ', vi*1.0/nTotalCols
        
        srRCol = pd.Series(index=arrDistinctUsers)
        lsRecords = dcDistinctVideos.get(v, None)
        assert (lsRecords is not None)
        for vid in lsRecords:
            record = (dfStreaming.iloc[vid])[['IMSI', 'RATIO']]
            srRCol.loc[record['IMSI']] = record['RATIO']
            
        # add to R
        dfR[v] = srRCol.fillna(0)
        
    
    # dfS
    dfStreaming.drop_duplicates(cols=lsVideoFeatures, inplace=True)
    del dfStreaming['IMSI']
    del dfStreaming['RATIO']
    
    return dfR.as_matrix(), dfStreaming.as_matrix()
        

def testOnSH():
    #===========================================================================
    # load data
    #===========================================================================
    dfStreaming = pd.read_csv('/media/data/playground/sh_xdr/test.out', sep='|', \
                              names= ['BEGIN_TIME','BEGIN_TIME_MSEL','MSISDN','IMSI',\
                                      'SERVER_IP','SERVER_PORT','APN','PROT_CATEGORY',\
                                      'PROT_TYPE','LAC','SAC','CI','IMEI','RAT','HOST',\
                                      'STREAMING_URL','STREAMING_FILESIZE','STREAMING_DW_PACKETS',\
                                      'STREAMING_DOWNLOAD_DELAY','ASSOCIATED_ID','L4_UL_THROUGHPUT',\
                                      'L4_DW_THROUGHPUT', 'INTBUFFER_FULL_FLAG', 'TCP_RTT', \
                                      'GET_STREAMING_DELAY', 'INTBUFFER_FULL_DELAY', 'SID', 'use_less'] )
    del dfStreaming['use_less']
    
    R, dfS = transform2VideoQualityMatrix(dfStreaming)
    assert(R.shape[1] == dfS.shape[0])
    
    return R, dfS
    
def transformNJData(strDataPath):
    '''
        transform NJ dataset into R, D, S matrices
        
        Note:
             1. before loading data, please manually replace all the 'none' with '' in
                original dataset. 
    '''
    
    #===========================================================================
    # load data set
    #===========================================================================
    dfData = pd.read_csv(strDataPath)
    
    #===========================================================================
    # clear data
    #===========================================================================
    # filter invalid rows out
    dfData = dfData[ (dfData['viewing time ratio']>=0.0) & (dfData['viewing time ratio']<=1.0) & \
                    (dfData['user MAC'] != np.nan) & (dfData['vid'] != np.nan) ]
    print ("%d rows have been loaded." % len(dfData) )
    
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
    # transform
    #===========================================================================
    R, D, S = transform2VideoQualityMatrixEx(dfData, lsColumns2Delete, dcColumns2Discretize, lsColumns2Vectorize, \
                                   strLabelColumnName, \
                                   strUserIDColumnName, lsUserProfileColumns, \
                                   strVideoIDColumnName, lsVideoQualityColumns)
    
    return R, D, S


if __name__ == '__main__':
    transformNJData("d:\\playground\\sh_xdr\\nj\\all_with_router_info.csv")