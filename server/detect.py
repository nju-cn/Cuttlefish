import torch
import cv2
import time
import datetime
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from utils.utils import *
from utils.datasets import *

class dataset(Dataset):
    def __init__(self,framelist):   #original frames
        self.framelist=np.zeros((len(framelist),3,416,416))
        for i in range(len(framelist)):
            frame=np.array(framelist[i])
            self.framelist[i]=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),(0,0,0),swapRB=True,crop=False)[0]
    def __len__(self):
        return len(self.framelist)
    def __getitem__(self, index):
        return self.framelist[index]

def detect_video(model,framelist,opt,size):
    
    starttime=time.time()
    print("start loading...")
    '''
    dataloader=DataLoader(
        dataset=dataset(framelist),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    '''
    a=np.array(framelist)
    bloblist=torch.from_numpy(cv2.dnn.blobFromImages(a,1/255.0,(416,416),(0,0,0),swapRB=True,crop=False))
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    bloblist=bloblist.type(Tensor)
    endtime=time.time()
    loadingtime=endtime-starttime
    print("loading time:",loadingtime)
    starttime=time.time()
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    #a = Variable(a.type(Tensor))
    print("\nPerforming object detection:")
    batchnum=math.ceil(len(bloblist)/opt.batch_size)
    for batch_i in range(batchnum):
        prev_time = time.time()
        input_imgs = Variable(bloblist[batch_i*opt.batch_size:(batch_i+1)*opt.batch_size])
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        for detection in detections:
            if detection is not None:
                boxes=[]
                #detection = rescale_boxes(detection, opt.img_size, (size[1],size[0]))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    x1=x1*size[0]/opt.img_size
                    x2=x2*size[0]/opt.img_size
                    y1=y1*size[1]/opt.img_size
                    y2=y2*size[1]/opt.img_size
                    boxes.append([int(x1),int(y1),int(x2-x1),int(y2-y1),int(cls_pred),cls_conf.item()])
                img_detections.append(boxes)
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
    endtime=time.time()
    detectingtime=endtime-starttime
    print("detecting time:",detectingtime)
    return img_detections,loadingtime
