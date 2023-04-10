from __future__ import division
import argparse
import torch
import cv2
import os
import numpy as np
from models import *
from utils.utils import *
from utils.datasets import *
from torchvision import models
from detect import detect_video
from myutils import evaluate,EvaluateVelocity
def prepare(videopath,timeslot=1.0):
    cap=cv2.VideoCapture(videopath)
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    slot_frame=int(fps*timeslot)
    success,frame=cap.read()
    index=0
    frames1080p=[]
    frames900p=[]
    frames720p=[]
    frames480p=[]
    frames360p=[]
    slots1080p=[]
    slots900p=[]
    slots720p=[]
    slots480p=[]
    slots360p=[]
    while success:
        frames1080p.append(cv2.resize(frame,(1920,1080)))
        frames900p.append(cv2.resize(frame,(1600,900)))
        frames720p.append(cv2.resize(frame,(1280,720)))
        frames480p.append(cv2.resize(frame,(720,480)))
        frames360p.append(cv2.resize(frame,(480,360)))
        index+=1
        success,frame=cap.read()
        if index==slot_frame:
            slots1080p.append(frames1080p)
            slots900p.append(frames900p)
            slots720p.append(frames720p)
            slots480p.append(frames480p)
            slots360p.append(frames360p)
            frames1080p=[]
            frames900p=[]
            frames720p=[]
            frames480p=[]
            frames360p=[]
            index=0
    return slots360p,slots480p,slots720p,slots900p,slots1080p

def pic_data(videopath,picpath,timeslot=1.0):
    model=models.vgg16(pretrained=True)
    cap=cv2.VideoCapture(videopath)
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    slot_frame=int(fps*timeslot)
    success,frame=cap.read()
    frame=cv2.dnn.blobFromImage(np.array(frame),1/255.0,(224,224),(0,0,0),swapRB=True,crop=False)
    fm=model(torch.from_numpy(frame))[0].detach().numpy()
    np.save(picpath+"0.npy",fm)
    index=0
    while success:
        index+=1
        if index%slot_frame==0:
            frame=cv2.dnn.blobFromImage(np.array(frame),1/255.0,(224,224),(0,0,0),swapRB=True,crop=False)
            fm=model(torch.from_numpy(frame))[0].detach().numpy()
            np.save(picpath+str(int(index/slot_frame))+".npy",fm)
        success,frame=cap.read()
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,required=True,help="path to input videos")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("start loading...")
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    print("finished!")
    model.eval()  # Set in evaluation mode
    print("start video preparing...")
    slots360p,slots480p,slots720p,slots900p,slots1080p=prepare(opt.input,1.0)
    print("finished!")
    datapath=opt.input+"_Data"
    if not os.path.exists(datapath):
        os.mkdir(datapath)
    picpath=datapath+"/feature_map/"
    os.mkdir(picpath)
    f1=open(datapath+"/1080p.txt",'w')
    f2=open(datapath+"/900p.txt",'w')
    f3=open(datapath+"/720p.txt",'w')
    f4=open(datapath+"/480p.txt",'w')
    f5=open(datapath+"/360p.txt",'w')
    filelist=[f1,f2,f3,f4,f5]

    for s5,s4,s3,s2,s1 in zip(slots360p,slots480p,slots720p,slots900p,slots1080p):
        box360p,time360p=detect_video(model,s5,opt,(480,360))
        box480p,time480p=detect_video(model,s4,opt,(720,480))
        box720p,time720p=detect_video(model,s3,opt,(1280,720))
        box900p,time900p=detect_video(model,s2,opt,(1600,900))
        box1080p,time1080p=detect_video(model,s1,opt,(1920,1080))
        r1 = evaluate(box1080p, box1080p, 1920/ 1920, 1080/ 1080, 0.8)
        r2 = evaluate(box1080p, box900p, 1920/ 1600.0, 1080/ 900.0, 0.8)
        r3 = evaluate(box1080p, box720p, 1920/ 1280.0, 1080/ 720.0, 0.8)
        r4 = evaluate(box1080p, box480p, 1920/ 720.0, 1080/ 480.0, 0.8)
        r5 = evaluate(box1080p, box360p, 1920/ 480.0, 1080/ 360.0, 0.8)
        v1= EvaluateVelocity(s1[0],s1[-1],box1080p[0],box1080p[-1],1.0)
        v2= EvaluateVelocity(s2[0],s2[-1],box900p[0],box900p[-1],1.0)
        v3= EvaluateVelocity(s3[0],s3[-1],box720p[0],box720p[-1],1.0)
        v4= EvaluateVelocity(s4[0],s4[-1],box480p[0],box480p[-1],1.0)
        v5= EvaluateVelocity(s5[0],s5[-1],box360p[0],box360p[-1],1.0)
        
        v=[v1,v2,v3,v4,v5]        
        r=[r1,r2,r3,r4,r5]
        t=[time1080p,time900p,time720p,time480p,time360p]
        for i in range(5):
            filelist[i].writelines(str(t[i])+"\n")
            filelist[i].writelines(str(r[i])+'\n')
            filelist[i].writelines(str(v[i])+"\n")
			
    pic_data(opt.input,picpath)
