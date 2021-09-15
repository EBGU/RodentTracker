import os
import numpy as np
import time
from multiprocessing import Pool
import psutil
import cv2
import matplotlib.pyplot as plt
import av #for better performance

##############################################################################
#For EPM, please select pionts from the OPEN arm to the CLOSE arm and press y:
#          o1
#       c3    c4
#          o2
#For OFT, please select pionts clockwise from upper left corner and press y:
#      UL1  UR2
#
#      LL4  LR3
#Press y to confirm remove background.
#For EPM please select the central neutral zone(four points, like OFT) and press y to confirm.
##############################################################################

######################
####Set Parameters####
######################
home = 'yourFolder'
src = home + '/Video'
tgt = home + '/Picture'
rmbg_tgt = home + '/Picture_rmbg'
logDir = home + '/log'
isEPM = False # whether EPM or OFT
startT = 60 # start at 60s
cropLen = 300 # crop only 300s(5min)
imgSize = 500 #resize Image
if isEPM:
    margin = 0.1 #for EPM, keep a margin of 10% image size
else:
    margin = 0.2 #for OFT, keep a margin of 20% image size
    useEllipse = False  #whether used ellipise to fit mouse, otherwise use 
refLenth = 100 # the arm lenth of EPM or size of OFT
centerCutOff = 0.5 # define the center zone, for OFT only!
multiThread = psutil.cpu_count(False)
video2img = False

img2binary = False
useAverFrame = True
cache = home + '/Cache'

tracking = True
preview = False
windowSize = 5 #window size for speed
Filter = 'aver' #a function to filter the positon, currently provide 'aver' 'median' 'none'
######################
##Function and Class##
######################
def padding(img):      #padding img in case rotate to the outside
    h, w = img.shape[:2]
    img_padded = np.zeros(shape=(w+h, w+h), dtype=np.uint8)
    img_padded[w//2:w//2+h,h//2:h//2+w] = img
    return img_padded

x = 0
vector = []
def mouse_img_cod(event, cod_x, cod_y, flags, param):
    global vector
    global x
    if event == cv2.EVENT_LBUTTONDOWN:
        if x == 0 :
            x += 1
            vector.append([cod_x,cod_y])
        else:
            x = 0
            vector.append([cod_x,cod_y])

class ImageCorrection():
    def __init__(self,refPoints,expand,half_size,EPM,crop=0.7):
        self.refPoints = refPoints
        self.center = half_size
        self.EPM = EPM
        self.crop = int(crop*self.center)
        if EPM:
            self.target = np.float32([[expand,self.center], [2*self.center-expand, self.center], [self.center, expand], [self.center, 2*self.center-expand]])
        else:
            self.target = np.float32([[expand,expand], [2*self.center-expand, expand], [2*self.center-expand, 2*self.center-expand], [expand, 2*self.center-expand]])
        self.M = cv2.getPerspectiveTransform(self.refPoints , self.target)
    def __call__(self,img):
        img = cv2.warpPerspective(img,self.M,(2*self.center,2*self.center))
        if self.EPM:
            img[0:self.crop,0:self.crop] = 255
            img[2*self.center-self.crop:2*self.center,0:self.crop] = 255
            img[2*self.center-self.crop:2*self.center,2*self.center-self.crop:2*self.center] = 255
            img[0:self.crop,2*self.center-self.crop:2*self.center] = 255
        return img

class ExtractAndWarp():
    def __init__(self,tgt,cache,startT,cropLen,expand=25,half_size=250,EPM = False,preview=False):
        self.tgt = tgt
        self.cache = cache
        self.startT =startT
        self.cropLen = cropLen
        self.expand =expand
        self.half_size =half_size
        self.EPM =EPM
        self.preview =preview
    def __call__(self,direction):
        fileAddr,vector = direction
        folder = os.path.join(self.tgt,fileAddr.split('.')[0].split('/')[-1])
        cache = os.path.join(self.cache,fileAddr.split('.')[0].split('/')[-1])+'.npy'
        try:
            os.mkdir(folder)
        except:
            pass
        warper = ImageCorrection(vector,self.expand,self.half_size,self.EPM)
        cap = cv2.VideoCapture(fileAddr)
        fps = cap.get(cv2.CAP_PROP_FPS)
        startAt = int( self.startT * fps) #in seconds
        #Record only 30min
        length = int(min((self.startT+self.cropLen) * fps,cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        container = av.open(fileAddr)
        for i,frame in enumerate(container.decode(video=0)):
            if i < np.ceil(fps*10):
                img = frame.to_ndarray(format='rgb24')
                img = warper(img)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)/ np.ceil(fps*10)
                try:
                    avgImg += img
                except:
                    avgImg = img
            if i >= startAt:
                img = frame.to_ndarray(format='rgb24')
                img = warper(img)
                if self.preview:
                    cv2.imshow("Image",img) 
                    k = cv2.waitKey(10) 
                    if k ==27:     # 键盘上Esc键的键值
                        cv2.destroyAllWindows() 
                        break
                else:
                    cv2.imwrite(os.path.join(folder,str(i-startAt+1)+'.jpg'), img,[cv2.IMWRITE_JPEG_QUALITY, 100])
            if i >= length:
                break
        np.save(cache,avgImg)
        container.close()
        return True

class frameAverage():
    def __init__(self,imgArray,dirs,nThread):        
        self.imgArray = imgArray
        self.windowSize = len(imgArray) // nThread + 1
        self.dirs = dirs
    #@timer
    def __call__(self,index):
        maxIndex = min(index+self.windowSize,len(self.imgArray))
        for path in self.imgArray[index:maxIndex]:
            img = cv2.imread(os.path.join(self.dirs,path), cv2.IMREAD_GRAYSCALE).astype(np.double)
            img = img / (maxIndex-index)
            try:
                avgImg += img
            except:
                avgImg = img
        return avgImg       

class rmBackground():

    def __init__(self,imgArray,dirs,src,tgt,background,nThread,threshold=25):        
        self.imgArray = imgArray
        self.windowSize = len(imgArray) // nThread + 1
        self.dirs = dirs
        self.background = background
        self.tgt = tgt
        self.src = src
        self.threshold =threshold
    #@timer
    def __call__(self,index):
        maxIndex = min(index+self.windowSize,len(self.imgArray))
        for path in self.imgArray[index:maxIndex]:
            img = cv2.imread(os.path.join(self.src,self.dirs,path), cv2.IMREAD_GRAYSCALE).astype(np.double)
            img = img - self.background 
            img[np.where(img<self.threshold)] = 0
            img = img.astype(np.uint8)
            img = cv2.medianBlur(img,5)
            img = 255-cv2.equalizeHist(img)
            img = cv2.medianBlur(img,5)
            cv2.imwrite(os.path.join(self.tgt,self.dirs,path), img)
        return True

class logger(object):
    def __init__(self,logDir):
        self.logDir = logDir
    def __call__(self,x,fileName):
        print(x)
        f = open(os.path.join(self.logDir,fileName+'.log'),'a')
        f.write(str(x)+'\n')
        f.close()

def trackingEPM(img,ori = None,kernal=5,thres = 150,preview=False): #kernel has to be odd
    result_gray=cv2.medianBlur(img, kernal)
    #result_binary = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,50) #use otsu autothreshold method
    ret,result_binary=cv2.threshold(result_gray,thres,255,0) 
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(255-result_binary, 4)
    largest = np.argmax(stats[:,4])
    stats[largest,4] = -1
    largest = np.argmax(stats[:,4])
    left = stats[largest,0]
    top = stats[largest,1]
    right = stats[largest,0]+stats[largest,2]
    down = stats[largest,1]+stats[largest,3]
    center = centroids[largest]
    if preview:
        fit = cv2.rectangle(ori, (left, top), (right, down), (255, 25, 25), 1)
        fit  = cv2.circle(fit, np.int32(center),3, (25, 25, 255), 1)
        cv2.imshow("Image",fit)
        k = cv2.waitKey(2)
        if k == 32:
            cv2.waitKey(0)
    return (left,right,top,down,center)

def trackingOFT(img,ori = None,kernal=11,thres = 100,preview=False):
    result_gray=cv2.medianBlur(img, kernal)
    #result_binary = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,50)
    ret,result_binary=cv2.threshold(result_gray,thres,255,0) #use otsu autothreshold method
    edge = cv2.Canny(result_binary,10,245)
    y,x=np.nonzero(edge)  #change coordination
    edge_list = np.array([[_x,_y] for _x,_y in zip(x,y)])  #list edge-points
    try:
        ellipse = cv2.fitEllipse(edge_list)  #  fit ellipse and return (x,y) as center,(2a，2b) as radius and angle
    except:
        ellipse = [(0,0),(0,0),1000]
    if preview:
        fit=cv2.ellipse(ori, ellipse, (255,25,25),1)
        cv2.imshow("Image",fit)
        cv2.waitKey(10)
    return ellipse

def Identity(x):
    return x[-1]

class Speedometer():
    def __init__(self,windowSize=5,Filter = 'aver'):
        self.container = []
        self.windowSize = windowSize
        self.filter = Filter
        assert(self.filter in ['aver','median','none'])
        self.speed = []
    def update(self,x):
        self.container.append(x)
        if len(self.container) == self.windowSize+2:
            if self.filter == 'aver':
                pastCord = np.mean(self.container[0:windowSize],axis=0)
                curentCord = np.mean(self.container[2:],axis=0)
            elif self.filter == 'median':
                pastCord = np.median(self.container[0:windowSize],axis=0)
                curentCord = np.median(self.container[2:],axis=0)
            elif self.filter == 'none':
                pastCord = self.container[windowSize//2+1]
                curentCord = self.container[windowSize//2+3]
            else:
                pass
            speed = ((pastCord[0]-curentCord[0])**2+(pastCord[1]-curentCord[1])**2)**0.5
            self.speed.append(speed)
            del(self.container[0])
            return speed
        else:
            return 0
    def aver(self):
        x = np.mean(self.speed)
        if np.isnan(x):
            return 0
        else:
            return x
######################
####Prepare images####
######################

if video2img:
    if os.path.isdir(src):
        try:
            os.mkdir(tgt)
        except:
            pass
        try:
            os.mkdir(logDir)
        except:
            pass
        try:
            os.mkdir(cache)
        except:
            pass
    else:
        raise ValueError('No video folder detected!')
    vList = os.listdir(src)
    direction=[]
    for v in vList:
        cap = cv2.VideoCapture(os.path.join(src,v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        startAt = startT * fps
        midFrame = int(min(cropLen * fps,cap.get(cv2.CAP_PROP_FRAME_COUNT)-startAt)) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES,startAt+midFrame)
        _,img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #img = padding(img)
        cv2.imshow("Image",img) 
        cv2.setMouseCallback("Image", mouse_img_cod) 
        k = cv2.waitKey(0) 
        if k ==121:     # press y
            cv2.destroyAllWindows() 
            cap.release()
        direction.append((os.path.join(src,v),np.float32(vector)))
        print((os.path.join(src,v),vector))
        vector = []
        print(len(direction))

    extractor = ExtractAndWarp(tgt,cache,startT,cropLen,expand=int(margin*imgSize*0.5),half_size=imgSize//2,EPM=isEPM,preview=False)
    for d in direction:
        extractor(d)

if img2binary:
    try: 
        os.mkdir(rmbg_tgt)
    except:
        pass
    dirList = os.listdir(tgt)

    for dirs in dirList:
        try:
            os.mkdir(os.path.join(rmbg_tgt,dirs))
        except:
            pass
        frameList = os.listdir(os.path.join(tgt,dirs))
        if useAverFrame:
            aver = frameAverage(frameList,os.path.join(tgt,dirs),multiThread)
            with Pool(multiThread) as p: 
                averaged=np.array(p.map(aver,range(0,len(frameList),aver.windowSize)))
            averaged = np.median(averaged,axis=0)
        else:
            averaged = np.load(os.path.join(cache,dirs)+'.npy')
        _averaged = averaged.astype(np.uint8)
        print(dirs)
        cv2.imshow('img',_averaged)
        k = cv2.waitKey(0)
        if k == 121: #121 is y
            cv2.destroyAllWindows()
            rmer = rmBackground(frameList,dirs,tgt,rmbg_tgt,averaged,multiThread)
            with Pool(multiThread) as p:
                p.map(rmer,range(0,len(frameList),rmer.windowSize))
printer = logger(logDir)
if tracking:
    print('Tracking! Ready? Go!')
    if isEPM:
        vList = os.listdir(src)
        for v in vList:
            speedo = Speedometer(windowSize=windowSize,Filter=Filter)
            cap = cv2.VideoCapture(os.path.join(src,v))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            localtime = time.asctime( time.localtime(time.time()) )
            v = v.split('.')[0]
            printer(localtime,v)
            printer('FPS = ' + str(fps),v)
            vector = []
            frameList = os.listdir(os.path.join(tgt,v))
            aver = frameAverage(frameList,os.path.join(tgt,v),multiThread)
            with Pool(multiThread) as p:
                averaged=np.array(p.map(aver,range(0,len(frameList),aver.windowSize)))
            averaged = np.median(averaged,axis=0)
            _averaged = averaged.astype(np.uint8)
            cv2.imshow('img',_averaged)
            cv2.setMouseCallback("img", mouse_img_cod) 
            k = cv2.waitKey(0) 
            if k ==121:     # press y
                cv2.destroyAllWindows() 
            printer('NeutralZone is:',v)
            printer(vector,v)
            printer('Time\tFrame\tleft\tright\ttop\tdown\tcenter_x\tcenter_y\tisOpen_center\tisOpen_any\tOpenTimeRatio_center\tOpenTimeRatio_any\tCurrentSpeed\tAverageSpeed',v)
            neutralL = np.min(np.array(vector)[:,0])
            neutralR = np.max(np.array(vector)[:,0])
            neutralT = np.min(np.array(vector)[:,1])
            neutralD = np.max(np.array(vector)[:,1])
            ioc = 0
            ioa = 1
            for i in range(len(frameList)):
                img = cv2.imread(os.path.join(rmbg_tgt,v,str(i+1)+'.jpg'),cv2.IMREAD_GRAYSCALE)
                ori = cv2.imread(os.path.join(tgt,v,str(i+1)+'.jpg'))
                left,right,top,down,(center_x,center_y) = trackingEPM(img,ori,preview=preview)
                speed = speedo.update([center_x,center_y])*fps*refLenth/(2*imgSize*(1-margin))
                averSpeed = speedo.aver()*fps*refLenth/(2*imgSize*(1-margin))
                if center_x <= neutralL or center_x >= neutralR:
                    isOpen_center = 1
                    ioc += 1
                else:
                    isOpen_center = 0
                if left <= neutralL or right >= neutralR:
                    isOpen_any = 1
                    ioa += 1
                else:
                    isOpen_any = 0
                printer('{:0>10.3f}\t{:0>6.0f}\t{:0>3.0f}\t{:0>3.0f}\t{:0>3.0f}\t{:0>3.0f}\t{:0>3.0f}\t{:0>3.0f}\t{:.0f}\t{:.0f}\t{:.5f}\t{:.5f}\t{:0>7.3f}\t{:0>7.3f}'.format((i+1)/fps,i+1,left,right,top,down,center_x,center_y,isOpen_center,isOpen_any,ioc/(i+1),ioa/(i+1),speed,averSpeed),v)
    else:
        vList = os.listdir(src)
        for v in vList:
            speedo = Speedometer(windowSize=windowSize,Filter=Filter)
            cap = cv2.VideoCapture(os.path.join(src,v))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            localtime = time.asctime( time.localtime(time.time()) )
            v = v.split('.')[0]
            printer(localtime,v)
            printer('FPS = ' + str(fps),v)
            printer('Time\tFrame\tcenter_x\tcenter_y\ta\tb\tangle\tcenter_distance\tisCenter\tCenterTimeRatio_center\tCurrentSpeed\tAverageSpeed',v)
            ic = 0
            frameList = os.listdir(os.path.join(tgt,v))
            for i in range(len(frameList)):
                img = cv2.imread(os.path.join(rmbg_tgt,v,str(i+1)+'.jpg'),cv2.IMREAD_GRAYSCALE)
                ori = cv2.imread(os.path.join(tgt,v,str(i+1)+'.jpg'))
                if useEllipse:
                    (center_x,center_y),(a,b),angle = trackingOFT(img,ori,preview=preview)
                else:
                    left,right,top,down,(center_x,center_y)= trackingEPM(img,ori,preview=preview)
                    a = right-left
                    b = down-top
                    angle = 0
                speed = speedo.update([center_x,center_y])*fps*refLenth/(2*imgSize*(1-margin))
                averSpeed = speedo.aver()*fps*refLenth/(2*imgSize*(1-margin))
                dis_x = abs(center_x-imgSize//2)
                dis_y = abs(center_y-imgSize//2)
                distance = ((dis_x**2+dis_y**2)**0.5)*refLenth/(imgSize*(1-margin))
                if max(dis_x,dis_y) < imgSize*0.5*(1-margin)*centerCutOff:
                    isCenter = 1
                    ic += 1
                else:
                    isCenter = 0
                printer('{:0>10.3f}\t{:0>6.0f}\t{:0>3.0f}\t{:0>3.0f}\t{:0>7.3f}\t{:0>7.3f}\t{:0>7.3f}\t{:0>7.3f}\t{:.0f}\t{:.5f}\t{:0>7.3f}\t{:0>7.3f}'.format((i+1)/fps,i+1,center_x,center_y,a,b,angle,distance,isCenter,ic/(i+1),speed,averSpeed),v)
            
            
          
