import sys
import os
import cv2
sys.path.append(r'../track/')
import socket
import numpy as np
import time
from ast import literal_eval

EDGE_CLOUD_ADDR=("114.212.85.50",30000)
MY_RECV_ADDR=("192.168.43.53",65530)

class SlotResult:
    def __init__(self,detecting_res=None,trans_latency=0.0,total_latency=0.0):
        #self.detecting_res=detecting_res
        self.trans_latency=trans_latency
        self.total_latency=total_latency
    def encode(self):
        res_str=""
        # if self.detecting_res is not None:
        #     for res in self.detecting_res:
        #         res_str+=res.convert2str()+","
        res_str=res_str+str(self.trans_latency)+","+str(self.total_latency)
        return res_str.encode()
    def decode(self,res_bytes):
        res_str=res_bytes.decode()
        info=res_str.split(',')
        print(info)
        self.trans_latency=float(info[-2])
        self.total_latency=float(info[-1])
        # self.detecting_res=[]
        # for i in range(0,len(info)-2):
        #     self.detecting_res.append(DetectResult().parse(info[i]))

# class DetectResult:
#     def __init__(self,x=0,y=0,h=0,w=0,conf=0.0,c=0):
#         self.x=x
#         self.y=y
#         self.h=h
#         self.w=w
#         self.conf=conf
#         self.c=c
#     def convert2str(self):
#         res_str=str(self.x)+';'+str(self.y)+';'+str(self.h)+';'+str(self.w)+';'+str(self.conf)+';'+str(self.c)
#         return res_str
#     def parse(self,res_str):
#         info=res_str.split(';')
#         self.x=int(info[0])
#         self.y=int(info[1])
#         self.h=int(info[2])
#         self.w=int(info[3])
#         self.conf=float(info[4])
#         self.c=int(info[5])


def upload_video(video_bytes):
    # global video_capture,totalslot
    # fps=action[0]
    # resolution=action[1]
    # if slot_index==0:
    #     video_capture=cv2.VideoCapture(video_name)
    #     totalslot=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))//int(video_capture.get(cv2.CAP_PROP_FPS))
    print("start connecting server...")
    sendsocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sendsocket.connect(EDGE_CLOUD_ADDR)
    #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    #send the video and fps
    #attrString=str(fps)+";"+str(size)+";"+video_name+";"+str(slot_index)
    start_time=time.time()
    headInfo="{'length':"+str(len(video_bytes))+",'timestamp':"+str(start_time)+"}"
    sendsocket.send(headInfo.encode())
    sendsocket.recv(1000)

    print("start uploading...")

    sendsocket.send(video_bytes)

    res_bytes=sendsocket.recv(99999).decode()
    sendsocket.close()

    print("get result from edge server")
    return float(res_bytes)


#print(upload_video(bytes(1000)))