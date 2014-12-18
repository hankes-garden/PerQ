# -*- coding: utf-8 -*-
'''
Description: 
    This module provides common functions

Created on 2014年8月25日

@author: jason
'''

import os
import struct
import socket
import pickle
import re
import datetime

def getFileList(strParentPath, strSuffix=None, strFileNamePattern=None):
    lsFiles = []
    for (dirpath, dirnames, filenames) in os.walk(strParentPath):
        for fn in sorted(filenames):

            # check prefix
            if strSuffix is not None:
                if fn.split('.')[-1] != strSuffix:
                    continue
                
            # check substring
            if strFileNamePattern is not None:
                if re.search(strFileNamePattern, fn) is None:
                    continue

            lsFiles.append(dirpath+'/'+fn)
    
    return lsFiles

def updateDictBySum(dc, key, newValue):
    if key in dc:
        dc[key] += newValue
    else:
        dc[key] = newValue
      
        
def ip2int(addr):                                                               
    return struct.unpack("!I", socket.inet_aton(addr))[0]                       

def int2ip(addr):                                                               
    return socket.inet_ntoa(struct.pack("!I", addr))  

def serialize2File(strOutFilePath, obj):
    if len(obj) != 0:
        with open(strOutFilePath, 'w') as hOutFile:
            pickle.dump(obj, hOutFile, protocol=-1)
    else:
        print("Nothing to serialize!")
   

def deserializeFromFile(strFilePath):
    obj = 0
    with open(strFilePath) as hFile:
        obj = pickle.load(hFile)
    return obj

def getSecondofDay(nTimestamp):
    dt = datetime.datetime.fromtimestamp(nTimestamp)
    return (dt.hour*3600 + dt.minute*60 + dt.second)
    
    
    
