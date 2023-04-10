#This file is aimed to match feature points of target objects in two frames.

import cv2
import track.object as object
import numpy as np
class MyFrame:
    def __init__(self,image):
        self.image=image
        self.size=image.shape[:2]
        self.gimage= cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.objectNumber=0
        self.objectlist=[]
        self.featurePoints=[]
        self.targetPoints=[]

    def initial_object(self,object_buffer,matched): #[(x1,y1,w1,h1,c1,confidence1),(x2,y2,w2,h2,c,confidence2),...]
        if matched:
            index=0
            for target_object in object_buffer:
                self.objectlist.append(object.object(target_object[0],target_object[1],target_object[2],target_object[3],index,target_object[4]))
                index+=1
        else:
            for target_object in object_buffer:
                self.objectlist.append(object.object(target_object[0],target_object[1],target_object[2],target_object[3],-1,target_object[4]))

    def featurePrepare(self):
        self.featurePoints=cv2.goodFeaturesToTrack(self.gimage,maxCorners=300,qualityLevel=0.01,minDistance=10)
        for i in range(len(self.featurePoints)):
            for object in self.objectlist:
                object.addfeaturepoint(self.featurePoints[i][0][0],self.featurePoints[i][0][1],i)

    def match(self,nextframe):
        nextframe.targetPoints,status,err=cv2.calcOpticalFlowPyrLK(self.gimage,nextframe.gimage,self.featurePoints,None, winSize=(20,20),maxLevel=3)
        for i in range(len(nextframe.targetPoints)):
            for object in nextframe.objectlist:
                object.addtargetpoint(nextframe.targetPoints[i][0][0],nextframe.targetPoints[i][0][1],i)

        for object in self.objectlist:
            object.match(nextframe.objectlist)


def track(lastframe,frame):
    lastframe.featurePrepare()
    lastframe.match(frame)
    
    total=0
    count=0
    for object in frame.objectlist:
        if object.distance!=(0,0):
            total+=object.distance[0]*100/lastframe.size[1]+object.distance[1]*100/lastframe.size[0]
            count+=1
    if count==0:
        avg_dis=0
    else:
        avg_dis=total/count
    print(avg_dis)
    return avg_dis




