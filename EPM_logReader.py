import numpy as np 
import cv2
import os
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as ks_test
from scipy import stats 

home = '/run/media/harold/Data_Storage/FanPu_Lab/0908_AVP1b_EPM' 
logDir = home + '/log'

fileList = os.listdir(logDir)

ctrlRatio = []
expRatio = []
maxLen = 0
for f in fileList:
    ratio = []
    with open(os.path.join(logDir,f)) as l:
        for i, lines in enumerate(l.readlines()):
            if i >=6:
                dist = float(lines.split('\t')[10])
                ratio.append(dist)   
    if not('_1b' in f):
        print(f)
        ctrlRatio.append(ratio)
    else:
        expRatio.append(ratio)
    if maxLen == 0:
        maxLen = len(ratio)
    else:
        maxLen = min(maxLen,len(ratio))

ctrlRatioNew = []
expRatioNew = []
for i in ctrlRatio:
    ctrlRatioNew.append(i[0:maxLen])
for i in expRatio:
    expRatioNew.append(i[0:maxLen])
ctrlRatioNew = np.array(ctrlRatioNew)
expRatioNew = np.array(expRatioNew)

t =np.arange(0,maxLen)

mean = np.mean(ctrlRatioNew,axis=0)
std = stats.sem(ctrlRatioNew,axis=0) # use 3 times sem 
plt.plot(t, mean,color="#0000FF",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#9999FF",alpha=0.5) 
#plt.show()


mean = np.mean(expRatioNew,axis=0)
std = stats.sem(expRatioNew,axis=0)  # use 3 times sem 
plt.plot(t, mean,color="#FF0000",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 

sig = []
for i in range(maxLen):
    try:
        p = stats.mannwhitneyu(ctrlRatioNew[:,i],expRatioNew[:,i])[1]
    except:
        p = 1
    sig.append(p)
    if p<0.05:
        print(ctrlRatioNew[:,i])
        print(expRatioNew[:,i])
        print(t[i],p)
sig = np.array(sig)
plt.fill_between(t, 0, 0.8, where=sig < 0.05,color='green', alpha=0.5)

plt.show()

print(maxLen)
print(ctrlRatioNew[:,maxLen-1])
print(expRatioNew[:,maxLen-1])