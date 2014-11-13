# -*- coding: utf-8 -*-
'''
Brief Description: 

@author: jason
'''

import common_function

g_dcUserVideos = {}
g_lsValidRecords = []
g_nRecordCount = 0
g_nValidRecord = 0
g_nInValidRecord = 0

def serializeTopUsers(dcUserRecords, nTop, strOutPath):
    '''
        This function writes the records of top users to disk
    '''
    lsSortedUserRecords = sorted(dcUserRecords.items(), key=lambda (k,v):len(v), reverse=True)
                        
    # write records of selected users to each file separately           
    for (k,v) in lsSortedUserRecords[:nTop]:
        strOutFilePath = ( (strOutPath) if strOutPath.endswith('/') else (strOutPath+'/') ) + ("%s.out" % k)
        with open (strOutFilePath, 'w') as hUserFile:
            hUserFile.writelines(v)
            
    return lsSortedUserRecords[:nTop]

def transform(strLine):
    lsItems = strLine.split('|')
    
    # change SERVER_IP to integer
    if (len(lsItems[4]) > 0):
        lsItems[4] = str(common_function.ip2int(lsItems[4]))
    
    # change LAC to integer
    if (len(lsItems[9]) > 0):
        lsItems[9] = int(lsItems[9], 16)
    
    # change SAC to integer
    if (len(lsItems[10]) > 0):
        lsItems[10] = int(lsItems[10], 16)
        
    # change Nan in INTBUFFER_FULL_FLAG to 1
    if (len(lsItems[22]) == 0):
        lsItems[22] = 1
    
    strTransformedLine = ''
    for i in lsItems:
        strItem = str(i)
        if (strItem != '\n'):
            strTransformedLine += (strItem + '|')
        else:
            strTransformedLine += strItem
        
    return strTransformedLine

def findValidRecords(strFilePath, g_dcUserVideos, hValidRecords=None):
    '''
        This function examine the key fields of each record,
        and store the valid ones to a dict, with key = IMSI
    '''
    global g_nRecordCount
    global g_nValidRecord
    with open(strFilePath, 'r') as hXDRFile:
        for line in hXDRFile:
            g_nRecordCount += 1
            lsItems = line.split('|')
            
            if (len(lsItems[3])>0 \
                 and len(lsItems[16])>0 \
                 and len(lsItems[17])>0 
                 and int(lsItems[16]) >= int(lsItems[17]) ): # valid record
                g_nValidRecord += 1
                
                # transform
                strTransformed = transform(line)
                
                # save to file
                updateUserRecords(strTransformed, g_dcUserVideos)
                if (hValidRecords is not None):
                    hValidRecords.write(strTransformed)
    print("%s is scanned." % strFilePath)
    
def countInvalid(strFilePath):
    '''
        This function examine the key fields of each record,
        and store the valid ones to a dict, with key = IMSI
    '''
    global g_nRecordCount
    global g_nValidRecord
    global g_nInValidRecord
    with open(strFilePath, 'r') as hXDRFile:
        for line in hXDRFile:
            g_nRecordCount += 1
            lsItems = line.split('|')
            
            if (len(lsItems[3])>0 \
                 and len(lsItems[16])>0 \
                 and len(lsItems[17])>0 ): # valid record
                g_nValidRecord += 1
                if (int(lsItems[16]) < int(lsItems[17]) ):
                    g_nInValidRecord += 1
    print("%s is scanned." % strFilePath)
            
            
def updateUserRecords(strLine, g_dcUserVideos):
    '''
        This function stores selected records by user id(IMSI)
    '''
    lsItems = strLine.split('|')
    lsRecords = g_dcUserVideos.get(lsItems[3], None)
    if (lsRecords is None):
        lsRecords = []
        g_dcUserVideos[lsItems[3]] = lsRecords
        
    lsRecords.append(strLine)
    
    
def execute(strXDRPath, strTopUserRecordsPath, strValidRecordPath, nTop=10):
    
    # get xdr list
    lsXDRFiles = common_function.getFileList(strXDRPath, 'dat')
    
    # open file to save all valid records
    with open(strValidRecordPath, 'w') as hValidFile:
        # find valid records
        for xdr in lsXDRFiles:
            findValidRecords(xdr, g_dcUserVideos, hValidFile)
        
    
    # select top user
    serializeTopUsers(g_dcUserVideos, nTop, strTopUserRecordsPath)
    
    # simple statistics
    nMax = 0
    nSum = 0
    for v in g_dcUserVideos.values():
        nVideo = len(v)
        nSum += nVideo
        if nVideo > nMax:
            nMax = nVideo
   
    print('#total video record: %d, valid ratio: %.2f' % (g_nRecordCount, g_nValidRecord*1.0/g_nRecordCount) )
    print("#user: %d" % len(g_dcUserVideos))
    print("#video: sum=%d, max=%d, avg=%.2f" % (nSum, nMax, nSum*1.0/len(g_dcUserVideos)) )
    
    return g_dcUserVideos
