import socket
import cv2
import numpy
import argparse
import torch
import time
import os
import queue
import threading
from models import *
from utils.utils import *
from utils.datasets import *
from detect import detect_video
from ast import literal_eval


class SlotResult:
    def __init__(self,detecting_res=None,trans_latency=0.0,total_latency=0.0):
        self.detecting_res=detecting_res
        self.trans_latency=trans_latency
        self.total_latency=total_latency
    def encode(self):
        res_str=""
        if self.detecting_res is not None:
            for res in self.detecting_res:
                res_str+=res.convert2str()+","
        res_str=res_str+str(self.trans_latency)+","+str(self.total_latency)
        return res_str.encode()
    def decode(self,res_bytes):
        res_str=res_bytes.decode()
        info=res_str.split(',')
        self.trans_latency=float(info[-2])
        self.total_latency=float(info[-1])
        self.detecting_res=[]
        for i in range(0,len(info)-2):
            self.detecting_res.append(DetectResult().parse(info[i]))

class DetectResult:
    def __init__(self,x=0,y=0,h=0,w=0,conf=0.0,c=0):
        self.x=x
        self.y=y
        self.h=h
        self.w=w
        self.conf=conf
        self.c=c
    def convert2str(self):
        res_str=str(self.x)+';'+str(self.y)+';'+str(self.h)+';'+str(self.w)+';'+str(self.conf)+';'+str(self.c)
        return res_str
    def parse(self,res_str):
        info=res_str.split(';')
        self.x=int(info[0])
        self.y=int(info[1])
        self.h=int(info[2])
        self.w=int(info[3])
        self.conf=float(info[4])
        self.c=int(info[5])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
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


    print("start loading yolo model...")
    # Set up model
    models_index={}
    models_index[0]=1080
    models_index[1]=900
    models_index[2]=720
    models_index[3]=480
    models_index[4]=360
    models={}
    for i in range(5):
        model = Darknet(opt.model_def, img_size=models_index[i]).to(device)

        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))
        print("finished!")
        model.eval()  # Set in evaluation mode
        models[models_index[i]]=model


    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1',30000))
    s.listen(5)
    while True:
        client_socket,addr=s.accept()
        headinfo=literal_eval(client_socket.recv(1000).decode())
        print(headinfo)
        length=headinfo['length']
        start_time=headinfo['timestamp']
        client_socket.send("ACK".encode())

        video_bytes = b''
        while len(video_bytes) < length:
            video_bytes += client_socket.recv(length - len(video_bytes))


        trans_latency=time.time()-start_time

        f=open('temp.264','wb')
        f.write(video_bytes)
        f.close()
        cap=cv2.VideoCapture('temp.264')
        frame_list=[]
        suc,f=cap.read()
        while suc is True:
            frame_list.append(f)
            suc,f=cap.read()
        cap.release()
        os.remove('temp.264')
        size=(frame_list[0].shape[1],frame_list[0].shape[0])
        box,_=detect_video(models[size[0]],frame_list,opt,size)

        r=DetectResult()
        detecting_res=[]
        detecting_res.append(r)

        total_latency=time.time()-start_time
        slot_res=SlotResult(detecting_res,trans_latency,total_latency)
        client_socket.send(slot_res.encode())
        client_socket.close()

