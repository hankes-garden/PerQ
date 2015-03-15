# -*- coding: utf-8 -*-

#===========================================================================
# This experiement does the following:
#   1. load data (include R, D, S, X, Y)
#   2. evaluate baseline
#   3. evaluate CMF
#===========================================================================

import pandas as pd
import numpy as np
import time
import tools.common_function as cf
import cmf.cmf_sgd as cmf
import model_comparison as mc
import data_processing.data2matrix as dm

if __name__ == '__main__':
    
    #===========================================================================
    # load data
    #===========================================================================
    print("loading data...")
#     dfR, dfD, dfS, dfX, srY, \
#     nVideoFeatureEnd, dcLoadingTrace = dm.transformSHData("../../data/user_201408.tsv", '../../data/video.csv', \
#                                                           dUserSamplingRatio=0.3, bTop=False, \
#                                                           lsUser2Select=None, bOnlyXY=False, \
#                                                           bFilterInvalid=True)
#     dfR = pd.read_pickle('D:\\playground\\personal_qoe\\data\\sh\\dfR_0discre_rand1000')
#     dfD = pd.read_pickle('D:\\playground\\personal_qoe\\data\\sh\\dfD_0discre_rand1000')
#     dfS = pd.read_pickle('D:\\playground\\personal_qoe\\data\\sh\\dfS_0discre_rand1000')
#     dfX = pd.read_pickle('D:\\playground\\personal_qoe\\data\\sh\\dfX_0discre_rand1000')
#     srY = pd.read_pickle('D:\\playground\\personal_qoe\\data\\sh\\srY_0discre_rand1000')
    
    dfR = pd.read_pickle('../../data/dfR_random30p')
    dfD = pd.read_pickle('../../data/dfD_random30p')
    dfS = pd.read_pickle('../../data/dfS_random30p')
    dfX = pd.read_pickle('../../data/dfX_random30p')
    srY = pd.read_pickle('../../data/srY_random30p')
    
    dcLoadingTrace = {}
    dcLoadingTrace['dm_result'] = 'read from pickle'
    
    # to save all intermediate result
    dcTrace = {}
    
    print("save info of transformed data")
    dcTrace.update(dcLoadingTrace)
    dcTrace['dm_tpRShape'] = dfR.shape
    dcTrace['dm_tpDShape'] = dfD.shape
    dcTrace['dm_tpSShape'] = dfS.shape
    dcTrace['dm_tpXShape'] = dfX.shape

    #===========================================================================
    # pickle to disk
    #===========================================================================
#     dfR.to_pickle('../../data/dfR_random30p')
#     dfD.to_pickle('../../data/dfD_random30p')
#     dfS.to_pickle('../../data/dfS_random30p')
#     dfX.to_pickle('../../data/dfX_random30p')
#     srY.to_pickle('../../data/srY_random30p')
    
    #===========================================================================
    # CMF
    #===========================================================================
    print("start to CMF evaluation...")
    # setup cmf
    arrAlphas_scaled = np.array([6, 20, 0.2])
    arrLambdas_scaled = np.array([2.0, 2.0, 2.0])
    f = 15
    nMaxStep = 500
    nFold = 10
    
    # init
    print('start to initialize...')
    R_reduced, D_reduced, S_reduced, \
    weightR_reduced, weightD_reduced, \
    weightS_reduced = cmf.init(dfR.as_matrix(), dfD.as_matrix(), dfS.as_matrix(), inplace=False, \
                               bReduceVideoDimension=True, dReductionRatio=0.7)

    
    # cross validation
    print("cmf cross validating...")
    dcCMFResult, lsBestTrainingRMSEs = cmf.crossValidate(R_reduced, D_reduced, S_reduced, \
                                                         weightR_reduced, weightD_reduced, weightS_reduced, \
                                                         arrAlphas_scaled, arrLambdas_scaled, \
                                                         f, nMaxStep, nFold, \
                                                         bDebugInfo=True)
    
    # save CMF result
    dcTrace['cmf_arrAlpahs'] = arrAlphas_scaled
    dcTrace['cmf_arrLambdas'] = arrLambdas_scaled
    dcTrace['cmf_f'] = f
    dcTrace['cmf_result'] = dcCMFResult
    dcTrace['cmf_lsBestTrainingRMSEs'] = lsBestTrainingRMSEs
    
   
    #===========================================================================
    # baseline
    #===========================================================================
    print("start to baseline evaluation...")
    # setup baseline
    strModelName = 'decision_tree_regression'
    modelParams = {'max_depth':4}
    nFold = 10

    dcBaselineResult = mc.baseline(dfX.as_matrix(), np.array(srY.tolist() ), strModelName, modelParams, \
                                   nFold, lsFeatureNames=dfX.columns.tolist() )
    
    # save baseline result
    dcTrace['baseline_name'] = strModelName
    dcTrace['baseline_params'] = modelParams
    dcTrace['baseline_result'] = dcBaselineResult
    
    # serialize result
    cf.serialize2File("../../data/dcTrace_random30p", dcTrace)
    
    print("****Congratulations! Experiment is finished.****")