import numpy as np 
import cv2
import os
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as ks_test
from scipy import stats 
import psutil
from multiprocessing import Pool

home = 'yourFolder' 
logDir = home + '/log'
bin_num =50
maxlen = 100
totalTime = 0
fileList = os.listdir(logDir)
multiThread = psutil.cpu_count(False)

ctrlBin = []
expBin = []

class logReader():
    def __init__(self,logDir):
        self.logDir = logDir
    def __call__(self,f):
        accDistr = []
        distanceM = []
        with open(os.path.join(logDir,f)) as l:
            for i, lines in enumerate(l.readlines()):
                if i >=3:
                    dist = np.abs(int(lines.split('\t')[2])-250)+np.abs(int(lines.split('\t')[3])-250)#float(lines.split('\t')[7])
                    distanceM.append(dist)   
                    accDistr.append(np.histogram(distanceM,bins=bin_num,range=(0,maxlen),density=True)[0])     
        return accDistr

Reader = logReader(logDir)
    
with Pool(multiThread) as p:
    Bin = p.map(Reader,fileList)

for i in Bin:
    if totalTime == 0:
        totalTime = len(i)
    else:
        totalTime = min(totalTime,len(i))

ctrlBin = []
expBin = []
for i,j in zip(Bin,fileList):
    if exp_keyword in j:
        print(j)
        expBin.append(i[0:totalTime])
    else:
        ctrlBin.append(i[0:totalTime])

ctrlBin = np.array(ctrlBin)
expBin = np.array(expBin)

class pValue():
    def __init__(self,bin_num,totalTime,expBin,ctrlBin):
        self.bin_num = bin_num
        self.totalTime = totalTime
        self.expBin = expBin
        self.ctrlBin = ctrlBin
    def __call__(self,x):
        i = x // self.bin_num
        j = x % self.bin_num
        try:
            p = stats.mannwhitneyu(ctrlBin[:,i,j],expBin[:,i,j])[1]
        except:
            p = 1  
        return p      
pCal = pValue(bin_num,totalTime,expBin,ctrlBin)

with Pool(multiThread) as p:
    pPlot = np.array(p.map(pCal,range(bin_num*totalTime))).reshape(totalTime,bin_num)


pPlot =np.where(pPlot<0.05,-np.log10(pPlot),0)



dist = np.linspace(0.0, maxlen, num=bin_num, endpoint=False)+maxlen*0.5/bin_num
t = np.arange(0,totalTime)
fig,(ax0,ax1,ax2,ax3) = plt.subplots(4,1)
expDist = np.mean(expBin,axis=0)
ctrlDist = np.mean(ctrlBin,axis=0)
im0 = ax0.pcolormesh(t, dist, np.mean(expBin,axis=0).transpose(),cmap='Reds')#,alpha=0.5)
im1 = ax1.pcolormesh(t, dist, np.mean(ctrlBin,axis=0).transpose(),cmap='Blues')#,alpha=0.5)
im2 = ax2.pcolormesh(t, dist, (np.mean(ctrlBin,axis=0)-np.mean(expBin,axis=0)).transpose(),cmap='turbo',vmax=0.005,vmin=-0.005)#,alpha=0.5)
fig.colorbar(im0, ax=ax0)
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
im3 = ax3.pcolormesh(t, dist,pPlot.transpose())
fig.colorbar(im3, ax=ax3)
plt.savefig(home+'/timeline.png',format='png',dpi=2000)
plt.show()


timePoint = int(input('Please choose a time point :'))


t = np.linspace(0.0, maxlen, num=bin_num, endpoint=False)+maxlen*0.5/bin_num

mean = np.mean(ctrlBin[:,timePoint,:],axis=0)
std = stats.sem(ctrlBin[:,timePoint,:],axis=0) # use 3 times sem 
plt.plot(t, mean,color="#0000FF",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#9999FF",alpha=0.5) 
#plt.show()


mean = np.mean(expBin[:,timePoint,:],axis=0)
std = stats.sem(expBin[:,timePoint,:],axis=0)  
plt.plot(t, mean,color="#FF0000",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 


sig = []
for i in range(bin_num):
    try:
        p = stats.mannwhitneyu(ctrlBin[:,timePoint,i],expBin[:,timePoint,i])[1]
    except:
        p = 1
    sig.append(p)
    if p<0.05:
        print(t[i],p)
sig = np.array(sig)
plt.fill_between(t, 0, 0.1, where=sig < 0.05,color='green', alpha=0.5)
plt.legend()
plt.show()
