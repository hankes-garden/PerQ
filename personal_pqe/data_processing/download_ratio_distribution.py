# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''

import numpy as np
import pandas as pd


if __name__ == '__main__':
    strVideoFilePath = "/mnt/disk1/yanglin/data/video.csv"
    
    dfData_video = pd.read_csv(strVideoFilePath, header=0, sep='\t')
    
    # add RATIO column
    dfData_video['ratio'] = dfData_video['streaming_dw_packets']*1.0/dfData_video['streaming_filesize']
    
    arrHist_dRatio, arrBin_dRatio = np.histogram(dfData_video['ratio'], bins=10, range=(0.0, 1.0 ) )
    print("arrBin_dRatio:")
    print arrBin_dRatio
    
    print("arrHist_dRatio:")
    print arrHist_dRatio
    