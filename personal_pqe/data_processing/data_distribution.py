# -*- coding: utf-8 -*-
'''
Description: 

@author: jason
'''

import pandas as pd
import matplotlib.pyplot as plt

def plotUserAgeDistribution(dfD):
    srAge = dfD[~dfD['age'].isnull()]['age'].astype(int)
    dcStat = {'age<20': 0, '20<=age<35':0, '35<=age<50':0, '50<=age':0}
    for k,v in srAge.iterkv():
        if (v<20):
            dcStat['age<20'] = dcStat['age<20'] + 1
        elif (v>=20 and v<35):
            dcStat['20<=age<35'] = dcStat['20<=age<35'] + 1
        elif (v>=35 and v<50):
            dcStat['35<=age<50'] = dcStat['35<=age<50'] + 1
        elif (v>=50):
            dcStat['50<=age'] = dcStat['50<=age'] + 1
        else:
            print('unknown value:%d' % v)
        
    srAgeDistrib = pd.Series(dcStat)
    srAgeDistrib = srAgeDistrib / 962.0
    
    srAgeDistrib.plot(kind='pie', autopct='%.2f%%', colors=['g', 'r', 'c', 'm'], labeldistance=1.05)
    plt.show()

if __name__ == '__main__':
    pass