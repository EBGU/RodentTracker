import os
import numpy as np
import time
from multiprocessing import Pool
import psutil
import cv2
import av #for better performance
from tqdm import tqdm
from copy import deepcopy
import shutil
def padding(img):      #padding img in case rotate to the outside
    h, w = img.shape[:2]
    img_padded = np.zeros(shape=(w+h, w+h), dtype=np.uint8)
    img_padded[w//2:w//2+h,h//2:h//2+w] = img
    return img_padded


class ImageCorrection():
    def __init__(self,refPoints,expand,half_size):
        self.refPoints = refPoints
        self.center = half_size
        self.target = np.float32([[expand,expand], [2*self.center[0]-expand, expand], [2*self.center[0]-expand, 2*self.center[1]-expand], [expand, 2*self.center[1]-expand]])
        self.M = cv2.getPerspectiveTransform(self.refPoints , self.target)
    def __call__(self,img):
        img = cv2.warpPerspective(img,self.M,(2*self.center[0],2*self.center[1]))
        return img

class ExtractAndWarp():
    def __init__(self,tgt,cache,startT,cropLen,expand=25,half_size=[200,300],preview=False):
        self.tgt = tgt
        self.cache = cache
        self.startT =startT
        self.cropLen = cropLen
        self.expand =expand
        self.half_size =half_size
        self.preview =preview
    def __call__(self,direction):
        fileAddr,vector = direction
        folder = os.path.join(self.tgt,os.path.basename(fileAddr).split('.')[0])
        cache = os.path.join(self.cache,os.path.basename(fileAddr).split('.')[0])+'.npy'
        try:
            os.mkdir(folder)
        except:
            pass
        warper = ImageCorrection(vector,self.expand,self.half_size)
        cap = cv2.VideoCapture(fileAddr)
        fps = cap.get(cv2.CAP_PROP_FPS)
        startAt = int( self.startT * fps) #in seconds
        #Record only 30min
        length = int(min((self.startT+self.cropLen) * fps,cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        container = av.open(fileAddr)
        print('Correcting image distortion, please wait...')
        for i,frame in tqdm(enumerate(container.decode(video=0))):
            if i < np.ceil(fps*10):
                img = frame.to_ndarray(format='rgb24')
                img = warper(img)
                #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)/ np.ceil(fps*10)
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
                    cv2.imwrite(os.path.join(folder,f'{(i-startAt+1):08d}.jpg'), img,[cv2.IMWRITE_JPEG_QUALITY, 100])
            if i >= length:
                break
        np.save(cache,avgImg)
        container.close()
        return folder,cache

class frameAverage():
    def __init__(self,imgArray,dirs,nThread):        
        self.imgArray = imgArray
        self.windowSize = len(imgArray) // nThread + 1
        self.dirs = dirs
    #@timer
    def __call__(self,index):
        maxIndex = min(index+self.windowSize,len(self.imgArray))
        if index == 0:
            imgs = tqdm(self.imgArray[index:maxIndex])
            print('Calculating backgrounds, please wait...')
        else:
            imgs = self.imgArray[index:maxIndex]
        for path in imgs:
            img = cv2.imread(os.path.join(self.dirs,path), cv2.IMREAD_GRAYSCALE).astype(np.double)
            img = img / (maxIndex-index)
            try:
                avgImg += img
            except:
                avgImg = img
        return avgImg     

class rmBackground():

    def __init__(self,imgArray,dirs,src,tgt,background,nThread,threshold=7):        
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
        if index == 0:
            imgs = tqdm(self.imgArray[index:maxIndex])
            print('Removing backgrounds, please wait...')
        else:
            imgs = self.imgArray[index:maxIndex]
        for path in imgs:
            img = cv2.imread(os.path.join(self.src,self.dirs,path), cv2.IMREAD_GRAYSCALE).astype(np.double)
            img = -img + self.background  #for black bg and white mice: img = img - self.background 
            img[np.where(img<self.threshold)] = 0
            img = img.astype(np.uint8)
            img = cv2.medianBlur(img,5)
            img = 255-cv2.equalizeHist(img)
            img = cv2.medianBlur(img,5)
            cv2.imwrite(os.path.join(self.tgt,self.dirs,path), img)
        return True


class logger(object):
    def __init__(self,filename):
        self.file = open(filename,'a')
    def __call__(self,x):
        #print(x)
        self.file.write(str(x)+'\n')
    def close(self):
        self.file.close()
def trackingEPM(img,ori = None,kernal=5,thres = 150,preview=False,exp=False,ctrl=False): #kernel has to be odd
    if exp:
        c = (25,255,25)
    elif ctrl:
        c = (25,25,255)
    else:
        c = (255,25,25)
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
        fit = cv2.rectangle(ori, (left, top), (right, down), c, 1)
        fit  = cv2.circle(fit, np.int32(center),3, c, 1)
        cv2.imshow("Image",fit)
        k = cv2.waitKey(1)
        if k == 32:
            cv2.waitKey(0)
    return (left,right,top,down,center)

def trackingOFT(img,ori = None,kernal=11,thres = 100,preview=False,exp=False,ctrl=False):
    if exp:
        c = (25,255,25)
    elif ctrl:
        c = (25,25,255)
    else:
        c = (255,25,25)
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
        fit=cv2.ellipse(ori, ellipse,c,1)
        cv2.imshow("Image",fit)
        cv2.waitKey(1)
    return ellipse


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
                pastCord = np.mean(self.container[0:self.windowSize],axis=0)
                curentCord = np.mean(self.container[2:],axis=0)
            elif self.filter == 'median':
                pastCord = np.median(self.container[0:self.windowSize],axis=0)
                curentCord = np.median(self.container[2:],axis=0)
            elif self.filter == 'none':
                pastCord = self.container[self.windowSize//2+1]
                curentCord = self.container[self.windowSize//2+3]
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
def ThreeChamber(
        videofile,
        tmp_dir,
        log_dir,
        startT = 10,
        cropLen = 600,
        imgSize = [600,400],
        margin = 0.0,
        useAverFrame = True, #if false, use first frames as background
        useEllipse = True,
        refLenth = 50,
        cup_radis = 3,
        accept_radis = 4,
        preview = True,
        windowSize = 5, #window size for speed
        Filter = 'aver', #a function to filter the positon, currently provide 'aver' 'median' 'none'
        delete_image = True,
        threshold = 7 #this is very sensitive and should be turned carefully 
):
    multiThread = psutil.cpu_count(False)
    cache = os.path.join(tmp_dir,'Cache')
    tgt = os.path.join(tmp_dir,'Picture')
    rmbg_tgt = os.path.join(tmp_dir,'Picture_rmbg') 
    try:
        os.mkdir(tgt)
    except:
        pass
    try:
        os.mkdir(cache)
    except:
        pass   
    try:
        os.mkdir(rmbg_tgt)
    except:
        pass
    vector = []
    def mouse_img_cod(event, cod_x, cod_y, flags, param):
        nonlocal vector
        if event == cv2.EVENT_LBUTTONDOWN:
                vector.append([cod_x,cod_y])
    
    print(f'processing {videofile}')
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    startAt = startT * fps
    midFrame = int(min(cropLen * fps,cap.get(cv2.CAP_PROP_FRAME_COUNT)-startAt)) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES,startAt+midFrame)
    _,img = cap.read()
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #img = padding(img)
    print('Please select INNER corners of the chamber with left clicks clockwisely from upper left corner, and then press Y')
    cv2.imshow("Image",img) 
    cv2.setMouseCallback("Image", mouse_img_cod) 
    k = cv2.waitKey(0) 
    if k ==121:     # press y
        cv2.destroyAllWindows() 
        cap.release()
    corners = deepcopy(np.float32(vector))
    
    #video2image
    extractor = ExtractAndWarp(tgt,cache,startT,cropLen,expand=int(margin*(imgSize[0]+imgSize[1])*0.25),half_size=(imgSize[0]//2,imgSize[1]//2),preview=False)
    extractor((videofile,corners))
    dirs = os.path.basename(videofile).split('.')[0]
    try:
        os.mkdir(os.path.join(rmbg_tgt,dirs))
    except:
        pass
    frameList = os.listdir(os.path.join(tgt,dirs))
    frameList.sort()
    if useAverFrame:
        aver = frameAverage(frameList,os.path.join(tgt,dirs),multiThread)
        with Pool(multiThread) as p: 
            averaged=np.array(p.map(aver,range(0,len(frameList),aver.windowSize)))
        averaged = np.median(averaged,axis=0)
    else:
        averaged = np.load(os.path.join(cache,dirs)+'.npy')
    _averaged = averaged.astype(np.uint8)
    print('Please select two cap CENTER from EXP to CTRL, and print Y to confirm the background.')
    vector = []
    cv2.imshow("Image",_averaged)
    cv2.setMouseCallback("Image", mouse_img_cod) 
    k = cv2.waitKey(0)
    if k == 121: #121 is y
        cv2.destroyAllWindows()
        EXP_center,Ctrl_center = deepcopy(vector)
        rmer = rmBackground(frameList,dirs,tgt,rmbg_tgt,averaged,multiThread,threshold=threshold)
        with Pool(multiThread) as p:
            p.map(rmer,range(0,len(frameList),rmer.windowSize))
    else:
        print('Exit due to unsatisfied background')
    
    #tracking
    printer = logger(os.path.join(log_dir,dirs+'.log'))
    print("Perform Tracking, please wait...")
    speedo = Speedometer(windowSize=windowSize,Filter=Filter)
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    localtime = time.asctime( time.localtime(time.time()) )
    printer(localtime)
    printer('FPS = ' + str(fps))
    printer('Time\tFrame\tcenter_x\tcenter_y\ta\tb\tangle\texp_distance\tEXPisCenter\tExpTimeRatio\tctrl_distance\tCTRLisCenter\tCtrlTimeRatio\tCurrentSpeed\tAverageSpeed')
    EXPic = 0
    CTRLic = 0
    frameList = os.listdir(os.path.join(tgt,dirs))
    frameList.sort()
    EXPisCenter = 0
    CTRLisCenter = 0
    for i,f in tqdm(list(enumerate(frameList))):
        img = cv2.imread(os.path.join(rmbg_tgt,dirs,f),cv2.IMREAD_GRAYSCALE)
        ori = cv2.imread(os.path.join(tgt,dirs,f))
        if useEllipse:
            (center_x,center_y),(a,b),angle = trackingOFT(img,ori,preview=preview,exp=EXPisCenter,ctrl=CTRLisCenter)
        else:
            left,right,top,down,(center_x,center_y)= trackingEPM(img,ori,preview=preview,exp=EXPisCenter,ctrl=CTRLisCenter)
            a = right-left
            b = down-top
            angle = 0
        speed = speedo.update([center_x,center_y])*fps*refLenth/((imgSize[0]+imgSize[1])*(1-margin))
        averSpeed = speedo.aver()*fps*refLenth/((imgSize[0]+imgSize[1])*(1-margin))
        exp_dis_x = abs(center_x-EXP_center[0])
        exp_dis_y = abs(center_y-EXP_center[1])
        ctrl_dis_x = abs(center_x-Ctrl_center[0])
        ctrl_dis_y = abs(center_y-Ctrl_center[1])
        exp_distance = ((exp_dis_x**2+exp_dis_y**2)**0.5)*refLenth/(0.5*(imgSize[0]+imgSize[1])*(1-margin))
        ctrl_distance = ((ctrl_dis_x**2+ctrl_dis_y**2)**0.5)*refLenth/(0.5*(imgSize[0]+imgSize[1])*(1-margin))
        if exp_distance>=cup_radis and exp_distance<=cup_radis+accept_radis:
            EXPisCenter = 1
            CTRLisCenter = 0
            EXPic += 1
        elif ctrl_distance>=cup_radis and ctrl_distance<=cup_radis+accept_radis:
            EXPisCenter = 0
            CTRLisCenter = 1
            CTRLic += 1        
        else:
            EXPisCenter = 0
            CTRLisCenter = 0

        printer('{:0>10.3f}\t{:0>6.0f}\t{:0>3.0f}\t{:0>3.0f}\t{:0>7.3f}\t{:0>7.3f}\t{:0>7.3f}\t{:0>7.3f}\t{:.0f}\t{:7d}\t{:0>7.3f}\t{:.0f}\t{:7d}\t{:0>7.3f}\t{:0>7.3f}'.format((i+1)/fps,i+1,center_x,center_y,a,b,angle,exp_distance,EXPisCenter,EXPic,ctrl_distance,CTRLisCenter,CTRLic,speed,averSpeed))
    printer.close()
    if delete_image:
        for f in os.listdir(tmp_dir):
            shutil.rmtree(os.path.join(tmp_dir,f))

if __name__ == "__main__":
    ThreeChamber(
        'VideoAbsolutePath',
        'TmpAbsolutePath',
        'LogAbsolutePath',
        delete_image=True
    )
