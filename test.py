import queue
import numpy as np
from ast import literal_eval
import os
import cv2
import socket
import time
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

        client_socket.send("123".encode())
        client_socket.close()

# import ctypes
# dll = ctypes.WinDLL("C:/Users/user/source/repos/Dlltest/x64/Release/Dlltest.dll") # 加载dll方式二
# print(dll.test_sum(1.1,3))

