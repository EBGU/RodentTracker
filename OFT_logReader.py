import numpy as np 
import cv2
import os
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as ks_test
from scipy import stats 

home = '/run/media/harold/Data_Storage/FanPu_Lab/AVP1b_OFT' 
logDir = home + '/log'
bin_num =50
maxlen = 100

fileList = os.listdir(logDir)

ctrlDist = []
expDist = []
ctrlBin = []
expBin = []
for f in fileList:
    distanceM = []
    with open(os.path.join(logDir,f)) as l:
        for i, lines in enumerate(l.readlines()):
            if i >=3:
                dist = float(lines.split('\t')[7])
                distanceM.append(dist)   
    if not('_1b' in f):
        print(f)
        ctrlDist.append(distanceM)
        ctrlBin.append(np.histogram(distanceM,bins=bin_num,range=(0,maxlen),density=True)[0])

    else:
        expDist.append(distanceM)
        expBin.append(np.histogram(distanceM,bins=bin_num,range=(0,maxlen),density=True)[0])


ctrlDist = np.concatenate(ctrlDist)
expDist = np.concatenate(expDist)
ctrlBin = np.array(ctrlBin)
expBin = np.array(expBin)


def violin(data):
    fig = plt.figure()
    ax2 = plt.subplot(111)
    parts = ax2.violinplot(data,widths=0.85,showmeans=False, showmedians=False,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    mid = np.array([np.percentile(x, [25, 50, 75]) for x in data]).transpose()
    means = [np.mean(x) for x in data]
    quartile1, medians, quartile3 = mid
    inds = np.arange(1, len(medians) + 1)
    ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax2.scatter(inds, means, marker='_', color='white', s=50, zorder=3)
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.show()
    return True



print(ks_test(ctrlDist,expDist))
print(stats.mannwhitneyu(ctrlDist,expDist))
violin([ctrlDist,expDist])


plt.hist(ctrlDist,bins=bin_num,density = True,cumulative=False,color='blue',alpha =0.5)
plt.hist(expDist,bins=bin_num,density = True,cumulative=False,color='red',alpha =0.5)
plt.show()

t = np.linspace(0.0, maxlen, num=bin_num, endpoint=False)+maxlen*0.5/bin_num

mean = np.mean(ctrlBin,axis=0)
std = stats.sem(ctrlBin,axis=0) # use 3 times sem 
plt.plot(t, mean,color="#0000FF",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#9999FF",alpha=0.5) 
#plt.show()


mean = np.mean(expBin,axis=0)
std = stats.sem(expBin,axis=0)  
plt.plot(t, mean,color="#FF0000",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 


sig = []
for i in range(bin_num):
    try:
        p = stats.mannwhitneyu(ctrlBin[:,i],expBin[:,i])[1]
    except:
        p = 1
    sig.append(p)
    if p<0.05:
        print(ctrlBin[:,i])
        print(expBin[:,i])
        print(t[i],p)
sig = np.array(sig)
plt.fill_between(t, 0, 0.06, where=sig < 0.05,color='green', alpha=0.5)
plt.legend()
plt.show()
