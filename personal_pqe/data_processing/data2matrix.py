# -*- coding: utf-8 -*-
'''
Brief Description: 
        This module transform XDR data records into sparse matrix
@author: jason
'''

import common_function as cf

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer



g_dcColumns2Discritize = {'BEGIN_TIME': [7*3600,9*3600,12*3600,14*3600,18*3600,20*3600], \
                          'STREAMING_FILESIZE': [10.0, 50.0, 100.0, 200.0, 300.0, 400.0],\
                          'STREAMING_DW_SPEED': [50.0, 100,0, 150.0, 200.0, 250.0, 300.0],\
                          'TCP_RTT': [500, 1000, 1500, 2000, 2500, 3000]
                          }

g_lsColumns2Digitalize = ['APN', 'PROT_TYPE', 'LOCATION', 'RAT', 'HOST']
g_dcDigitMappingTable = {}

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
    
        

def transform2VideoQualityMatrix(df):
    '''
        This function transform xdr data into video-matrix quality
        1. discretize continuous data
        2. vectorize categorical data
        3. transfrom into R, S matrices
         
    '''
   
    #===========================================================================
    # 
    # columns = ['BEGIN_TIME','BEGIN_TIME_MSEL','MSISDN','IMSI',\
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
    
    # DW_SPEEDvalid_records_new
    dfStreaming['STREAMING_DOWNLOAD_DELAY'].replace(0,1, inplace=True)
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
        if srColumn.name == 'BEGIN_TIME':
            func = cf.getSecondofDay
        arrCuts = discretizeColumn(srColumn, func)
        if arrCuts is not None:
            dfStreaming[col] = arrCuts
        
        # digitalize categorical data
        
        arrDigits, dcMappingTable = digitalizeColumn(srColumn)
        if arrDigits is not None:
            dfStreaming[col] = arrDigits
            g_dcDigitMappingTable[srColumn.name] = dcMappingTable
            
            
            
     #==========================================================================
     # ['BEGIN_TIME',
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
    # construct matrices S, R
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
        
    
    # S
    dfStreaming.drop_duplicates(cols=lsVideoFeatures, inplace=True)
    del dfStreaming['IMSI']
    del dfStreaming['RATIO']
    
    return dfR.as_matrix(), dfStreaming.as_matrix()
        

if __name__ == '__main__':
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
    
    R, S = transform2VideoQualityMatrix(dfStreaming)
    assert(R.shape[1] == S.shape[0])