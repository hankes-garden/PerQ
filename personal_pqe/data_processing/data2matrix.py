# -*- coding: utf-8 -*-
'''
Brief Description: 
        This module transform XDR data records into sparse matrix
@author: jason
'''

import pandas as pd
import numpy as np
import sklearn.preprocessing as prepro
import gc
import random

g_dcColumns2Discritize = {'BEGIN_TIME': [7*3600,9*3600,12*3600,14*3600,18*3600,20*3600], \
                          'STREAMING_FILESIZE': [10.0, 50.0, 100.0, 200.0, 300.0, 400.0],\
                          'STREAMING_DW_SPEED': [50.0, 100,0, 150.0, 200.0, 250.0, 300.0],\
                          'TCP_RTT': [500, 1000, 1500, 2000, 2500, 3000]
                          }

g_lsColumns2Digitalize = ['APN', 'PROT_TYPE', 'LOCATION', 'RAT', 'HOST']
g_dcDigitMappingTable = {}

def discretizeColumnEx(srColumn, lsBins, func=None):
    '''
        discretize value based on predefined bins,
        if none is given, then cut by 25%, 50%, 75%
    '''
    strName = srColumn.name
    if lsBins is None: # cut according to its distribution
        sStat = srColumn.describe()
        lsBins = [sStat['25%'], sStat['50%'], sStat['75%'] ]
    else:
        if func is not None:
            srColumn = srColumn.apply(func)
        
    lsBins.append(srColumn.min() )
    lsBins.append(srColumn.max() )
    lsBins = np.unique(lsBins).tolist()
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
    arrUniqueValues, arrIndex = np.unique(srColumn.fillna(value=-1), return_inverse=True)
    
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


def transformSHData(strUserFilePath, strVideoFilePath, nTotalUser2Sample, bTop, lsUser2Select=None):
    '''
        This function transform shanghai data set to R, S, D matrices.
        Namely, this function does the following tasks:
        1. load data;
        2. add columns;
        3. setup transform rules;
        4. transform data set to matrices;
        
        params:
                strUserFilePath - user data file path
                strVideoFilePath - video data file path
                nTotalUser2Sample - total number of user to sample
                bTop - sample top N users
                lsUser2Select - specify user to select manually
        
        returns:
                dfR, dfD, dfS - dataframe of R, D, S
                dfFlattenTable - concatenated user and video features, can be used for baseline evaluation
                sFlattenLabel - labels corresponds to dfFlattenTable
        Note:
            user features (v1.0) = 
                ['localbase_outer_call_dur', 'ld_call_dur', 'roam_call_dur', \
                'localbase_called_dur', 'ld_called_dur', 'roam_called_dur', \
                'cm_dur', 'ct_dur', 'busy_call_dur', 'fest_call_dur', 'sms_p2p_mo_cnt', \
                'sms_p2p_inter_mo_cnt', 'sms_p2p_inner_mo_cnt', 'sms_p2p_other_mo_cnt', \
                'sms_p2p_cm_mo_cnt', 'sms_p2p_ct_mo_cnt', 'sms_info_mo_cnt', \
                'sms_p2p_roam_int_mo_cnt', 'mms_p2p_mo_cnt', 'mms_p2p_inter_mo_cnt', \
                'mms_p2p_inner_mo_cnt', 'mms_p2p_other_mo_cnt', 'mms_p2p_cm_mo_cnt', \
                'mms_p2p_ct_mo_cnt', 'mms_p2p_roam_int_mo_cnt', 'all_call_cnt', 'voice_cnt', \
                'local_base_call_cnt', 'ld_call_cnt', 'roam_call_cnt', 'caller_cnt', 'voice_dur', \
                'caller_dur', 'localbase_inner_call_dur', 'free_call_dur', 'call_10010_cnt', \
                'call_10010_manual_cnt', 'sms_bill_cnt', 'sms_p2p_mt_cnt', 'mms_cnt', \
                'mms_p2p_mt_cnt', 'gprs_all_flux', 'gender', 'age', 'credit_value', \
                'innet_dura', 'total_charge', 'gprs_flux', 'gprs_charge', 'local_call_minutes', \
                'toll_call_minutes', 'roam_call_minutes', 'voice_call_minutes', 'p2p_sms_mo_cnt', \
                'p2p_sms_mo_charge', 'pspt_type', 'is_shanghai', 'town_id', 'sale_id', 'pagerank', \
                'product_id', 'product_price', 'product_knd', 'gift_voice_call_dur', \
                'gift_sms_mo_cnt', 'gift_flux_value', 'distinct_serve_count', 'serve_sms_count', \
                'lp', 'balance', 'balance_diff']
        
            video attributes = 
                ['begin_time', 'begin_time_msel', 'imsi', 'server_ip', \
                 'server_port', 'apn', 'prot_category', 'prot_type', \
                 'lac', 'sac', 'ci', 'imei', 'rat', 'host', 'streaming_url', \
                 'streaming_filesize', 'streaming_dw_packets', 'streaming_download_delay', \
                 'associated_id', 'l4_ul_throughput', 'l4_dw_throughput', 'intbuffer_full_flag', \
                 'tcp_rtt', 'get_streaming_delay', 'intbuffer_full_delay', 'sid', 'date_partition']
            
            user features (v2.0) =      
            ['msisdn',
             'age',
             'gender',
             'innet_dura',
             'last_mon_bill_fee',
             'last_mon_web_flux',
             'last_mon_web_visit_cnt',
             'last_mon_web_user_hit_cnt',
             'last_mon_all_gn_flux',
             'last_mon_gn_visit_cnt',
             'ov_up_pptv_flag',
             'ov_up_youku_flag',
             'ov_up_sohuvidio_flag',
             'ov_up_sohuvidio_hq_flag',
             'gprs_all_flux',
             'gprs_net_flux',
             'gprs_net_up_flux',
             'gprs_net_down_flux',
             'gprs_wap_flux',
             'gprs_wap_up_flux',
             'gprs_wap_down_flux',
             'gprs_net_cnt',
             'gprs_wap_cnt',
             'g3_gprs_up_b',
             'g3_gprs_down_b',
             'ov_up_gprs_flux',
             'gprs_flux',
             'gprs_charge',
             'gift_flux_value']
            
            
    '''
    #===========================================================================
    # load data set (no index is used!)
    #===========================================================================
    dfData_user = pd.read_csv(strUserFilePath, header=0, sep='\t')
    dfData_video = pd.read_csv(strVideoFilePath, header=0, sep='\t')
    
    #===========================================================================
    # add columns
    #===========================================================================
    # add RATIO column
    dfData_video['ratio'] = dfData_video['streaming_dw_packets']*1.0/dfData_video['streaming_filesize']
    
    # add LOCATION column
    dfData_video['location'] = dfData_video['lac'].apply(str) + '-' \
                                + dfData_video['sac'].apply(str)\
                                + '-' + dfData_video['ci'].apply(str)
    
    # add DW_SPEED column
    dfData_video['streaming_download_delay'].replace(0, 1, inplace=True)
    dfData_video['streaming_download_speed'] = \
        dfData_video['streaming_dw_packets']*1.0/dfData_video['streaming_download_delay']
        
    #===========================================================================
    # setup transform rules
    #===========================================================================
    #----user features----
    strIDColumnName_user = "msisdn"
#     lsColumns2Delete_user = ['free_call_dur','roam_call_minutes']
#     dcColumns2Discretize_user = {}
#     lsColumns2Vectorize_user = ['town_id', 'sale_id', 'product_id']
    lsColumns2Delete_user = ['last_mon_web_user_hit_cnt', 'ov_up_pptv_flag', \
                             'ov_up_sohuvidio_flag', 'ov_up_sohuvidio_hq_flag']
    dcColumns2Discretize_user = {}
    lsColumns2Vectorize_user = []
    
    #----video----
    strIDColumnName_video = "streaming_url"
    
    lsColumns2Delete_video = ['begin_time_msel', 'imsi', 'server_ip', 'server_port', 'imei',\
                              'streaming_dw_packets', 'associated_id', 'sid',\
                              'lac', 'sac', 'ci', 'intbuffer_full_delay', 'location', \
                              'host', 'streaming_download_delay', 'intbuffer_full_flag', \
                              'get_streaming_delay']
    
    dcColumns2Discretize_video = {'begin_time': [7*3600,9*3600,12*3600,14*3600,18*3600,20*3600], \
                                 'streaming_filesize': [10.0, 50.0, 100.0, 200.0], \
                                 'streaming_download_speed': [50.0, 100,0, 150.0, 200.0, 250.0, 300.0], \
                                 'l4_ul_throughput':[1000.0, 2000.0, 4000.0], \
                                 'l4_dw_throughput':[500.0, 1000.0, 2000.0, 3000.0], \
                                 'tcp_rtt': [500, 1000, 1500, 2000, 2500, 3000]}
    
    lsColumns2Vectorize_video = ['prot_category', 'prot_type', 'apn', \
                                 'date_partition']
    
    #----ratio matrix----
    strLabelColumnName = "ratio"
    
    #===========================================================================
    # transform
    #===========================================================================
    return transform2Matrices(dfData_user, strIDColumnName_user, \
                              lsColumns2Delete_user, dcColumns2Discretize_user, lsColumns2Vectorize_user, \
                              dfData_video, strIDColumnName_video, \
                              lsColumns2Delete_video, dcColumns2Discretize_video, lsColumns2Vectorize_video, \
                              strLabelColumnName, nTotalUser2Sample, bTop, lsUser2Select)
    

def transform2mt(dfData_user, strIDColumnName_user, \
                 lsColumns2Delete_user, dcColumns2Discretize_user, lsColumns2Vectorize_user, \
                 dfData_video, strIDColumnName_video, \
                 lsColumns2Delete_video, dcColumns2Discretize_video, lsColumns2Vectorize_video, \
                 strLabelColumnName, lsUser2Select):
    '''
        Given two dataframes which contains user feature and video attribute data,
        this function transform them to R, D, S matrices.
        Namely, this function does the following works:
        0. filter out invalid tuples;
        1. delete useless columns;
        2. discretize continuous data;
        3. use OneHotEncoder to map categorical data;
        4. transform into R, D, S;
        
        param:
                as their name described.
                
        returns:
                R    - matrix which use np.uint8(255) to represent missing values
                D, S - matrices which use NAN to represent missing values
    '''
    
    #===========================================================================
    # filter out invalid tuples
    #===========================================================================
    print("start to filter out invalid tuples...")
    lsMasks_user = (~dfData_user[strIDColumnName_user].isnull() )
    lsMasks_video = (~dfData_video[strIDColumnName_user].isnull() ) \
                     & (~dfData_video['streaming_dw_packets'].isnull() ) \
                     & (~dfData_video['streaming_filesize'].isnull() ) \
                     & (dfData_video['streaming_dw_packets']<=dfData_video['streaming_filesize'])
                     
    dfData_user = dfData_user[lsMasks_user]
    dfData_video = dfData_video[lsMasks_video]
    
    print("-->valid data size: %d users, %d records" % (len(dfData_user), len(dfData_video) ) )
    
    #===========================================================================
    # find common users
    #===========================================================================
    print("start to find common users...")
    # change both of them into string, in case of the wrong type cast of pandas
    dfData_user[strIDColumnName_user] = dfData_user[strIDColumnName_user].astype(str).str[:11]
    dfData_video[strIDColumnName_user] = dfData_video[strIDColumnName_user].astype(str).str[:11]

    lsCommonUsers = list(\
                         set(dfData_user[strIDColumnName_user].tolist())    \
                         & set(dfData_video[strIDColumnName_user].tolist()) \
                         )
    
    if (lsUser2Select is not None):
        print('start to select top %d users from common users...' % len(lsUser2Select) )
        lsCommonUsers = list( set(lsCommonUsers) & set(lsUser2Select) )
        
    print("-->%d users co-exist in both user and video data set." % len(lsCommonUsers) )
    dfData_user = dfData_user[dfData_user[strIDColumnName_user].isin(lsCommonUsers)]
    dfData_video = dfData_video[dfData_video[strIDColumnName_user].isin(lsCommonUsers)]
    
    #===========================================================================
    # delete useless columns
    #===========================================================================
    print("start to delete useless columns...")
    #----user----
    for strColName in lsColumns2Delete_user:
        del dfData_user[strColName]
        
    #----video----
    for strColName in lsColumns2Delete_video:
        del dfData_video[strColName]
    
    #===========================================================================
    # time to reduce memory usage
    #===========================================================================
    gc.collect()
    
    #===========================================================================
    # discretize continuous data
    #===========================================================================
    print("start to discretize continuous data...")
    #----user----
    for (strColName,lsBins) in dcColumns2Discretize_user.iteritems():
        sCol = dfData_user[strColName]

        arrCuts = discretizeColumnEx(sCol, lsBins, None)
        if arrCuts is not None:
            dfData_user[strColName] = arrCuts # replace, no need to delete original one
        else:
            print ("discretize user column:%s failed." % strColName)
            
    #----video----
    for (strColName,lsBins) in dcColumns2Discretize_video.iteritems():
        sCol = dfData_video[strColName]

        arrCuts = discretizeColumnEx(sCol, lsBins, None)
        if arrCuts is not None:
            dfData_video[strColName] = arrCuts
        else:
            print ("discretize video column:%s failed." % strColName)
        
    
    #===========================================================================
    # mapping categorical data
    #===========================================================================
    print("start to mapping categorical data...")
    
    #----user----
    for strColName in lsColumns2Vectorize_user:
        sCol = dfData_user[strColName]
        print("mapping %s, unique value:%d..." % (strColName, len(sCol.unique() ) ) )
        # digitalize categorical data
        arrDigits, dcMappingTable = digitalizeColumnEx(sCol)
        g_dcDigitMappingTable[sCol.name] = dcMappingTable
        
        # vectorize
        enc = prepro.OneHotEncoder()
        mtEncoded = enc.fit_transform(pd.DataFrame(arrDigits) ).toarray()
        
        # add mapped columns
        for ci in xrange(mtEncoded.shape[1]):
            strName = "%s_%d" % (strColName, ci)
            dfData_user[strName] = mtEncoded[:,ci]
        
        # delete original column
        del dfData_user[strColName]
        
    #----video----    
    for strColName in lsColumns2Vectorize_video:
        sCol = dfData_video[strColName]
        print("mapping %s, unique value:%d..." % (strColName, len(sCol.unique() ) ) )
        # digitalize categorical data
        arrDigits, dcMappingTable = digitalizeColumnEx(sCol)
        g_dcDigitMappingTable[sCol.name] = dcMappingTable
        
        # vectorize
        enc = prepro.OneHotEncoder()
        mtEncoded = enc.fit_transform(pd.DataFrame(arrDigits) ).toarray()
        
        # add mapped columns
        for ci in xrange(mtEncoded.shape[1]):
            strName = "%s_%d" % (strColName, ci)
            dfData_video[strName] = mtEncoded[:,ci]
        
        # delete original column
        del dfData_video[strColName]
        
    # TODO: find way to mapping it back!
    
    #===========================================================================
    # time to reduce memory usage
    #===========================================================================
    gc.collect()
    
    #===========================================================================
    # transfrom into D
    #===========================================================================
    print("start to transfrom into D...")
    dfD = dfData_user.drop_duplicates(strIDColumnName_user)
    
    lsUserOrder = dfD[strIDColumnName_user].tolist()
    
    #===========================================================================
    # transform into S
    #===========================================================================
    print("start to transform into S...")
    dcS ={}
    dcUsers2VideoLabels = {} # to remember labels attached to each video 
    nCount = 0
    for ind, sRow in dfData_video.iterrows():
        if (nCount % 1000 == 0):
            print("-->%.2f%%" % (nCount*100.0/len(dfData_video)) )
            
        sVideoRecord = sRow.drop([strIDColumnName_video, strIDColumnName_user, strLabelColumnName]) 
        strVid = ''.join([str(i)+',' for i in sVideoRecord.values])    # only use video qualities as key
        strUid = sRow[strIDColumnName_user]      # user of current row
        dLabel = sRow[strLabelColumnName]    # label of current row
        
        # update dfS
        dcS[strVid] = sVideoRecord
        
        # prepare for R (row: vid, column:uid)
        dcVideos2Labels = dcUsers2VideoLabels.get(strUid)
        if dcVideos2Labels is None: # if already exist, then add
            dcVideos2Labels = {}
            dcUsers2VideoLabels[strUid] = dcVideos2Labels
        
        # FIXME: what if the same user has watched two videos(same video twice or two videos of same quality)
        dcVideos2Labels[strVid] = np.uint8(dLabel*100.0)
        nCount += 1
    
    dfS = pd.DataFrame(dcS).T.convert_objects(convert_numeric = True)
    
    #===========================================================================
    # time to reduce memory usage
    #===========================================================================
    gc.collect()
    
    #===========================================================================
    # transform into R
    #===========================================================================
    print("start to transform into R...")
    lsFrames = []
    nStart = 0
    nSliceSize = 5000
    nEnd = nStart + nSliceSize
    
    lsUsers2VideoLabels = dcUsers2VideoLabels.items()
    nTotalSize = len(lsUsers2VideoLabels)
    
    while (nStart < nTotalSize ):
        dc = {k:v for (k,v) in lsUsers2VideoLabels[nStart:nEnd]}
        df = pd.DataFrame(dc, columns=lsUserOrder).fillna(np.uint8(255)).astype(np.uint8)
        lsFrames.append(df)

        nStart = nEnd
        nEnd = nStart + nSliceSize
        if (nEnd > nTotalSize):
            nEnd = nTotalSize
            
        print('-->R: %.2f%%' % (nEnd*100.0/nTotalSize) )
    
    #===========================================================================
    # time to reduce memory usage
    #===========================================================================
    gc.collect()
    
    #===========================================================================
    # aggregate to R
    #===========================================================================
    print('start to aggregate R...')
    # remember order
    lsVideoOrder_R = []
    for f in lsFrames:
        lsVideoOrder_R += f.index.tolist()
    
    # aggregate
    dfR = pd.concat(lsFrames, ignore_index=True)
    
    lsUserOrder_R = dfR.columns
    
    dfR = dfR.T # change to user-video matrix
    dfR.columns = lsVideoOrder_R # set vid to R
    
    #===========================================================================
    # sort w.r.t R
    #===========================================================================
    print('start to sort w.r.t R...')
    dfD = dfD.set_index(strIDColumnName_user)
    dfD = dfD.reindex(lsUserOrder_R)
    
    dfS = dfS.reindex(lsVideoOrder_R)
    
    print("Congratulations! transformation is finished.")
    
    return lsFrames, lsUserOrder_R, lsVideoOrder_R, dfR, dfD, dfS

def AggregateR(lsFrames, lsUserOrder, lsVideoOrder):
    #===========================================================================
    # contact into dfR
    #===========================================================================
    print ('merging into R...')
    
    nCount = 0
    dfR = pd.DataFrame(index=lsUserOrder)
    for f in lsFrames:
        dfR = dfR.merge(f, left_index=True, right_index=True, how='outer', copy=False)
        
        nCount += 1
        print("-->%.2f%%" % (nCount*100.0/len(lsFrames)) )
        
        gc.collect()
    
    #===========================================================================
    # time to reduce memory usage
    #===========================================================================
    gc.collect()
    
    #===========================================================================
    # sort to ensure R, D, S are in the same order
    #===========================================================================
    dfR = dfR[lsVideoOrder]
    
    print('Aggregation of R is finished, shape:%d, %d' % (dfR.shape[0], dfR.shape[1]) )
    
    return dfR.as_matrix()

def transform2Matrices(dfData_user, strIDColumnName_user, \
                       lsColumns2Delete_user, dcColumns2Discretize_user, lsColumns2Vectorize_user, \
                       dfData_video, strIDColumnName_video, \
                       lsColumns2Delete_video, dcColumns2Discretize_video, lsColumns2Vectorize_video, \
                       strLabelColumnName, \
                       nTotalUser2Sample, bTop, lsUser2Select=None):
    '''
        Given two data frames which contains user feature and video feature data, this function 
        transform them to R, D, S matrices.
        Specifically, this function does the following works:
        0. filter out invalid tuples;
        1. delete useless columns;
        2. do NOT discretize continuous data (no need!)
        3. use OneHotEncoder to map categorical data;
        4. transform into R, D, S;
        
        param:
                dfData_user, strIDColumnName_user,
                lsColumns2Delete_user, dcColumns2Discretize_user, lsColumns2Vectorize_user, \
                dfData_video, strIDColumnName_video, \
                lsColumns2Delete_video, dcColumns2Discretize_video, lsColumns2Vectorize_video, \
                strLabelColumnName,

                nTotalUser2Sample - total number of user to sample
                bTop - sample top N users
                lsUser2Select - specify user to select manually
                
        returns:
                R, D, S - matrix which use np.nan to represent missing values
    '''
    
    #===========================================================================
    # filter out invalid tuples
    #===========================================================================
    print("start to filter out invalid tuples...")
    lsMasks_user = (~dfData_user[strIDColumnName_user].isnull() )
    lsMasks_video = (~dfData_video[strIDColumnName_user].isnull() ) \
                     & (~dfData_video['streaming_dw_packets'].isnull() ) \
                     & (~dfData_video['streaming_filesize'].isnull() ) \
                     & (dfData_video['streaming_dw_packets']<=dfData_video['streaming_filesize'])
                     
    dfData_user = dfData_user[lsMasks_user]
    dfData_video = dfData_video[lsMasks_video]
    
    print("-->valid data size: %d users, %d records" % (len(dfData_user), len(dfData_video) ) )
    
    #===========================================================================
    # find common users
    #===========================================================================
    print("start to find common users...")
    # change both of them into string, in case of the wrong type cast of pandas
    dfData_user[strIDColumnName_user] = dfData_user[strIDColumnName_user].astype(str).str[:11]
    dfData_video[strIDColumnName_user] = dfData_video[strIDColumnName_user].astype(str).str[:11]

    lsCommonUsers = list(\
                         set(dfData_user[strIDColumnName_user].tolist())    \
                         & set(dfData_video[strIDColumnName_user].tolist()) \
                         )
    
    print("-->%d users co-exist in both user and video data set." % len(lsCommonUsers) )
    
    if (lsUser2Select is not None):
        print('start to select %d users according to given list...' % len(lsUser2Select) )
        lsCommonUsers = list( set(lsCommonUsers) & set(lsUser2Select) )
    
    # only use tuples of these selected users
    dfData_user = dfData_user[ dfData_user[strIDColumnName_user].isin(lsCommonUsers) ]
    dfData_video = dfData_video[ dfData_video[strIDColumnName_user].isin(lsCommonUsers) ]
    
    #===========================================================================
    # sample records
    #===========================================================================
    if (nTotalUser2Sample is not None):
        lsUser2Sample = None
        if (bTop):
            srUserRank = dfData_video[strIDColumnName_user].value_counts(sort=True, ascending=False)
            lsUser2Sample = (srUserRank.index.tolist())[:nTotalUser2Sample]
        else:
            lsUser2Sample = random.sample(dfData_video[strIDColumnName_user].unique(), nTotalUser2Sample)
        
        # only use tuples of these selected users
        print('start to sample %d from %s users....' % (nTotalUser2Sample, ('top' if bTop else 'random') ) )
        dfData_user = dfData_user[ dfData_user[strIDColumnName_user].isin(lsUser2Sample) ]
        dfData_video = dfData_video[ dfData_video[strIDColumnName_user].isin(lsUser2Sample) ]
    
    #===========================================================================
    # delete useless columns
    #===========================================================================
    print("start to delete useless columns...")
    #----user----
    for strColName in lsColumns2Delete_user:
        del dfData_user[strColName]
        
    #----video----
    for strColName in lsColumns2Delete_video:
        del dfData_video[strColName]
    
    #===========================================================================
    # mapping categorical data
    #===========================================================================
    print("start to mapping categorical data...")
    
    #----user----
    for strColName in lsColumns2Vectorize_user:
        sCol = dfData_user[strColName]
        print("mapping %s, unique value:%d..." % (strColName, len(sCol.unique() ) ) )
        # digitalize categorical data
        arrDigits, dcMappingTable = digitalizeColumnEx(sCol)
        g_dcDigitMappingTable[sCol.name] = dcMappingTable
        
        # vectorize
        enc = prepro.OneHotEncoder()
        mtEncoded = enc.fit_transform(pd.DataFrame(arrDigits) ).toarray()
        
        # add mapped columns
        for ci in xrange(mtEncoded.shape[1]):
            strName = "%s_%d" % (strColName, ci)
            dfData_user[strName] = mtEncoded[:,ci]
        
        # delete original column
        del dfData_user[strColName]
        
    #----video----    
    for strColName in lsColumns2Vectorize_video:
        sCol = dfData_video[strColName]
        print("mapping %s, unique value:%d..." % (strColName, len(sCol.unique() ) ) )
        # digitalize categorical data
        arrDigits, dcMappingTable = digitalizeColumnEx(sCol)
        g_dcDigitMappingTable[sCol.name] = dcMappingTable
        
        # vectorize
        enc = prepro.OneHotEncoder()
        mtEncoded = enc.fit_transform(pd.DataFrame(arrDigits) ).toarray()
        
        # add mapped columns
        for ci in xrange(mtEncoded.shape[1]):
            strName = "%s_%d" % (strColName, ci)
            dfData_video[strName] = mtEncoded[:,ci]
        
        # delete original column
        del dfData_video[strColName]
        
    # TODO: find way to mapping it back!
    
    #===========================================================================
    # time to reduce memory usage
    #===========================================================================
    gc.collect()
    
    #===========================================================================
    # transform 2 flatten table
    #===========================================================================
    print('start to transform into flatten table...')
    dfX = pd.merge(dfData_video, dfData_user, how='inner', \
                              on=strIDColumnName_user, copy=True)
    srY = dfData_video[strLabelColumnName]
    del dfX[strLabelColumnName]
    del dfX[strIDColumnName_user]
    del dfX[strIDColumnName_video]
    
    #===========================================================================
    # transfrom into D (still include userID for now)
    #===========================================================================
    print("start to transfrom into D...")
    dfD = dfData_user.drop_duplicates(strIDColumnName_user)
    lsUserOrder = dfD[strIDColumnName_user].tolist()
    
    #===========================================================================
    # transform into R
    # Note: the idea here is that, since we use video feature as an identifier
    #       of videos, each video records will be treated as unique ''video'',
    #       because the possibility that two video records have same features
    #       is so small.
    #       This could be a problem as it would lead us to a R matrix in which 
    #       there is only one known ratio in each column. NEED TO THINK IT AGAIN!   
    #===========================================================================
    print("start to transform into R...")
    dfR = pd.DataFrame(index=lsUserOrder)
    for ind, row in dfData_video.iterrows():
        strUid = row.loc[strIDColumnName_user]
        dRatio = row.loc[strLabelColumnName]
        sCol = pd.Series(index=lsUserOrder)
        sCol.loc[strUid] = dRatio
        
        dfR[ind] = sCol # use index as new ''vid''
    
    #===========================================================================
    # time to reduce memory usage
    #===========================================================================
    gc.collect()
    
    #===========================================================================
    # transform into S
    #===========================================================================
    print("start to transform into S...")
    del dfData_video[strIDColumnName_video]
    del dfData_video[strIDColumnName_user]
    del dfData_video[strLabelColumnName]
    
    dfS = dfData_video
    
    lsVideoOrder = dfS.index.tolist()
    
   
    #===========================================================================
    # sort to ensure these matrices are in some order
    #===========================================================================
    print('start to sort w.r.t R...')
    
    # no need to sort S any more
    
    # sort D
    dfD = dfD.set_index(strIDColumnName_user) # this also get rid of user ID
    dfD = dfD.reindex(lsUserOrder)
    
    # sort R
    dfR = dfR[lsVideoOrder]
    dfR = dfR.reindex(lsUserOrder)
    
    print("Congratulations! transformation is finished.")
    
    return dfR, dfD, dfS, dfX, srY

