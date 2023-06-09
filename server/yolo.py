from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image=Image.open("dining_table.jpg")
    img=np.array(image)
    img=torch.from_numpy(img)
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

        # Extract labels
    #labels += targets[:, 1].tolist()
        # Rescale target
    #targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    #targets[:, 2:] *= img_size

    #imgs = Variable(imgs.type(Tensor), requires_grad=False)

    with torch.no_grad():
        outputs = model(img)
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
    #sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    print("start loading...")
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    print("finished")

    #precision, recall, AP, f1, ap_class = evaluate(
        #model,
        #path=valid_path,
        #iou_thres=opt.iou_thres,
        #conf_thres=opt.conf_thres,
        #nms_thres=opt.nms_thres,
        #img_size=opt.img_size,
        #batch_size=8,
    #)
    evaluate(model,
        iou_thres=opt.iou_thres,conf_thres=opt.conf_thres,nms_thres=opt.nms_thres,
        img_size=opt.img_size,batch_size=8)

    #print("Average Precisions:")
    #for i, c in enumerate(ap_class):
        #print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    #print(f"mAP: {AP.mean()}")
