from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class mydataset(Dataset):
    def __init__(self,size,video_path):
        self.framelist=[]
        cap=cv2.VideoCapture(video_path)
        success,frame=cap.read()
        while success:
            frame=cv2.resize(frame,size)
            frame=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),(0,0,0),swapRB=True,crop=False)
            self.framelist.append(np.array(frame)[0])
            success,frame=cap.read()
    def __len__(self):
        return len(self.framelist)
    def __getitem__(self, index):
        return self.framelist[index]

def evaluate(boxes_list0, boxes_list1, x_times, y_times, confidence):
    # x_times = 2: width -> 2 * width
    # y_times = 2: height -> 2 * height
    frame_num = len(boxes_list0)
    frame_index = 0
    scores = 0
    for frame_index in range(frame_num):
        TP_FN = len(boxes_list0[frame_index])
        TP_FP = len(boxes_list1[frame_index])
        
        TP = 0
        for box1 in boxes_list1[frame_index]:
            for box0 in boxes_list0[frame_index]:
                if similarGet(box1[0] * x_times, box1[1] * y_times, box1[2] * x_times, box1[3] * y_times, box0[0], box0[1], box0[2], box0[3]) >= confidence:
                    TP += 1
                    break
        if TP != 0:
            scores += 2/(TP_FN/TP + TP_FP/TP)
        
    average_score = scores / frame_num    
    return average_score

def detect_video(model,video_path,size):
    starttime=time.time()
    dataloader=DataLoader(
        dataset=mydataset(size,video_path),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    endtime=time.time()
    print("loading time:",endtime-starttime)
    print("finished!")
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    #a = Variable(a.type(Tensor))
    print("\nPerforming object detection:")
    for batch_i, input_imgs in enumerate(dataloader):
        prev_time = time.time()
        input_imgs = Variable(input_imgs.type(Tensor))
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        for detection in detections:
            if detection is not None:         
                boxes=[]
                #detection = rescale_boxes(detection, opt.img_size, (size[1],size[0]))   
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    x1=x1*size[0]/opt.img_size
                    x2=x2*size[0]/opt.img_size
                    y1=y1*size[1]/opt.img_size
                    y2=y2*size[1]/opt.img_size
                    boxes.append([int(x1),int(y1),int(x2-x1),int(y2-y1),classes[int(cls_pred)],cls_conf.item()])
                img_detections.append(boxes)
    return img_detections

def evaluate(boxes_list0, boxes_list1, x_times, y_times, confidence):
    # x_times = 2: width -> 2 * width
    # y_times = 2: height -> 2 * height
    frame_num = len(boxes_list0)
    frame_index = 0
    scores = 0
    for frame_index in range(frame_num):
        TP_FN = len(boxes_list0[frame_index])
        TP_FP = len(boxes_list1[frame_index])
        
        TP = 0
        for box1 in boxes_list1[frame_index]:
            for box0 in boxes_list0[frame_index]:
                if similarGet(box1[0] * x_times, box1[1] * y_times, box1[2] * x_times, box1[3] * y_times, box0[0], box0[1], box0[2], box0[3]) >= confidence:
                    TP += 1
                    break
        if TP != 0:
            scores += 2/(TP_FN/TP + TP_FP/TP)
        
    average_score = scores / frame_num    
    return average_score

def similarGet(x1,y1,w1,h1,x2,y2,w2,h2):
    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + w1, x2 + w2)
    ymax = min(y1 + h1, y2 + h2)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return 0
    cross_square = width * height
    union_square = w1 * h1 + w2 * h2 - cross_square
    return cross_square / union_square

def getResult(box1080p, box900p, box720p, box540p, box360p, confidence):
    r1 = evaluate(box1080p, box1080p, 1920/ 1920, 1080/ 1080, confidence)
    r2 = evaluate(box1080p, box900p, 1920/ 1600.0, 1080/ 900.0, confidence)
    r3 = evaluate(box1080p, box720p, 1920/ 1280.0, 1080/ 720.0, confidence)
    r4 = evaluate(box1080p, box540p, 1920/ 960.0, 1080/ 540.0, confidence)
    r5 = evaluate(box1080p, box360p, 1920/ 480.0, 1080/ 360.0, confidence)
    return r1, r2, r3, r4, r5

if __name__ == "__main__":
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
    
    box1080p=detect_video(model,opt.input,(1920,1080))
    box900p=detect_video(model,opt.input,(1600,900))
    box720p=detect_video(model,opt.input,(1280,720))
    box540p=detect_video(model,opt.input,(960,540))
    box360p=detect_video(model,opt.input,(480,360))
    confidenceList = [0.7, 0.8, 0.9,0.95]
    for c in confidenceList:
        r1,r2,r3,r4,r5 = getResult(box1080p, box900p, box720p, box540p, box360p, c)
        print("result for threshold " + str(c) + ":")
        print(r1)
        print(r2)
        print(r3)
        print(r4)
        print(r5)
    '''
    total=0
    for detections in img_detections:
        if detections is not None:
            print(len(detections))
            total+=len(detections)
        # Rescale boxes to original image
        #detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        #unique_labels = detections[:, -1].cpu().unique()
        #n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds)
            #print("image:")    
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_w = x2 - x1
                box_h = y2 - y1
                #print(int(x1),int(y1),int(box_w),int(box_h),"Class: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
    print("total:",total)
    '''
    '''
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, input_imgs in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        print(input_imgs.size())
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        img_detections.extend(detections)


    # Iterate through images and save plot of detections
    
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        
        # Create plot
        img = np.array(Image.open(path))
        #plt.figure()
        #fig, ax = plt.subplots(1)
        #ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_w = x2 - x1
                box_h = y2 - y1
                print(int(x1),int(y1),int(box_w),int(box_h),"Class: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                #bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                #ax.add_patch(bbox)
                # Add label
                #plt.text(
                 #   x1,
                 #   y1,
                 #   s=classes[int(cls_pred)],
                 #   color="white",
                 #   verticalalignment="top",
                 #   bbox={"color": color, "pad": 0},
                #)
        '''
        # Save generated image with detections
        #plt.axis("off")
        #plt.gca().xaxis.set_major_locator(NullLocator())
        #plt.gca().yaxis.set_major_locator(NullLocator())
        #filename = path.split("/")[-1].split(".")[0]
        #plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        #plt.close()
