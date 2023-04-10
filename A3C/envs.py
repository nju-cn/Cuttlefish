import numpy as np
import random
import math
import time
import torch
import sys
import cv2
# import feature
# import object
from video.upload import upload_video
from video.encoder import encode
# import multiprocessing as mp
from torchvision import models

'''
class DetectResult:
    def __init__(self,x=0,y=0,h=0,w=0,conf=0,c=0):
        self.x=x
        self.y=y
        self.h=h
        self.w=0
        self.conf=0
        self.c=0
'''


class TrainEnv:
    def __init__(self, train_path_name="pedestrian.avi", slot_time=1.0, test=False):

        # Params of training
        self.TRAIN_DATA_ROOT_PATH = "../traindata/"
        self.train_path_name = train_path_name
        self.train_data_file = []
        self.trainset_info = None
        self.test = test
        self.tracefile = None
        # Params of env

        self.slot_time = slot_time
        self.max_fps = 30
        self.min_fps = 5
        self.max_v = 20.0
        self.resolution_list = ["360p", "480p", "720p", "900p", "1080p"]
        self.N_S = 4  # historical fps,historical resolution, estimated bandwidth,velocity
        self.A_S = int((self.max_fps - self.min_fps + 1) * len(self.resolution_list))
        self.state = None
        self.images_state = None

        self.slot_index = 0
        self.bandwidth_buffer = None
        self.action_buffer = None
        self.weight = [0.5, 0.25, 0.15, 0.1]
        self.episode_length = 200

        # paramter
        self.a1 = 0.35
        self.a2 = 0.30
        self.a3 = 0.30
        self.bias = self.a1 * 0.7 - self.a2 * 0.75 - self.a3 * 0.88

    def StateActiondim(self):
        return self.N_S, self.A_S

    def reset(self):
        if self.test:
            self.tracefile = open("../traces/bandwidth/" + "0.log", 'r')
        else:
            self.tracefile = open("../traces/bandwidth/" + str(random.randint(0, 32)) + ".log", 'r')
        bandwidth = self.getBandwidth()
        self.bandwidth_buffer = [bandwidth, bandwidth, bandwidth, bandwidth]

        min_conf = (self.min_fps, 0)
        self.action_buffer = [min_conf, min_conf, min_conf, min_conf]
        # initial state
        self.state = []
        self.state.append(np.array([conf[0] for conf in self.action_buffer], np.float32))
        self.state.append(np.array([conf[1] for conf in self.action_buffer], np.float32))
        self.state.append(bandwidth)
        self.state.append(1.0)

        # self.images_state = self.get_image_state()
        self.images_state = None
        if self.slot_index == 0:
            info_file = open(self.TRAIN_DATA_ROOT_PATH + self.train_path_name + "/info.txt", 'r')
            bitrate = int(info_file.readline())
            max_slot_count = int(info_file.readline())
            self.trainset_info = (bitrate, max_slot_count)
            for f in self.train_data_file:
                f.close()
            self.train_data_file = []
            self.train_data_file.append(open(self.TRAIN_DATA_ROOT_PATH + self.train_path_name + "/360p.txt", 'r'))
            self.train_data_file.append(open(self.TRAIN_DATA_ROOT_PATH + self.train_path_name + "/480p.txt", 'r'))
            self.train_data_file.append(open(self.TRAIN_DATA_ROOT_PATH + self.train_path_name + "/720p.txt", 'r'))
            self.train_data_file.append(open(self.TRAIN_DATA_ROOT_PATH + self.train_path_name + "/900p.txt", 'r'))
            self.train_data_file.append(open(self.TRAIN_DATA_ROOT_PATH + self.train_path_name + "/1080p.txt", 'r'))
        return self.state, self.images_state

    def step(self, action):
        fps_index = action % (self.max_fps - self.min_fps + 1)
        fps = fps_index + self.min_fps
        resolution_index = action // (self.max_fps - self.min_fps + 1)
        resolution = self.resolution_list[resolution_index]

        # Simulate uploading
        t, accuracy, v = self.ReadRessultFile(resolution_index)
        # Update state
        self.action_buffer = [(fps_index, resolution_index), self.action_buffer[0], self.action_buffer[1],
                              self.action_buffer[2]]
        self.state[0] = np.array([conf[0] for conf in self.action_buffer], np.float32)
        self.state[1] = np.array([conf[1] for conf in self.action_buffer], np.float32)
        done = False
        bandwidth = self.getBandwidth()
        self.bandwidth_buffer = [bandwidth, self.bandwidth_buffer[0], self.bandwidth_buffer[1],
                                 self.bandwidth_buffer[2]]
        self.state[2] = self.estimateBandwidth()
        self.state[3] = v
        size = (0, 0)
        if resolution_index == 0:
            size = (480, 360)
        elif resolution_index == 1:
            size = (720, 480)
        elif resolution_index == 2:
            size = (1280, 720)
        elif resolution_index == 3:
            size = (1600, 900)
        elif resolution_index == 4:
            size = (1920, 1080)
        delay = (t + self.trainset_info[0] * size[0] * size[1] / (
                1920 * 1080 * bandwidth * 1000)) * fps / self.max_fps + 0.6 * fps / self.max_fps
        self.slot_index += 1
        self.images_state = self.get_image_state()
        done = (self.slot_index % self.episode_length == 0)
        if done:
            if self.slot_index + self.episode_length > self.trainset_info[1]:
                self.slot_index = 0
        reward, r1, r2, r3 = self.estimateReward(delay, accuracy, fps, v)
        return self.state, self.images_state, reward, done, r1, r2, r3

    def argmax_step(self, actionlist):
        res = [-100, 0, 0, 0]
        info = self.ReadRessultFile()
        bandwidth = self.getBandwidth()
        for actionindex in actionlist:
            fps = int((self.fps_range[0] + actionindex % (self.fps_range[1] - self.fps_range[0] + 1)) / self.slot)
            resolution_index = int(actionindex // (self.fps_range[1] - self.fps_range[0] + 1))
            if resolution_index == 0:
                size = (480, 360)
                t, accuracy, v = info[0]
            elif resolution_index == 1:
                size = (720, 480)
                t, accuracy, v = info[1]
            elif resolution_index == 2:
                size = (1280, 720)
                t, accuracy, v = info[2]
            elif resolution_index == 3:
                size = (1600, 900)
                t, accuracy, v = info[3]
            elif resolution_index == 4:
                size = (1920, 1080)
                t, accuracy, v = info[4]
            delay = ((t + self.trainset_info[0] * size[0] * size[1] / (
                    1920 * 1080 * bandwidth * 1000)) * fps / self.max_fps + 0.6 * fps / self.max_fps + 1) / 2
            reward, r1, r2, r3 = self.estimateReward(delay, accuracy, fps, v)
            if reward > res[0]:
                self.state[3] = v
                self.state[0:2] = [fps, resolution_index]
                res = [reward, r1, r2, r3]
        self.slot_index += 1
        self.images_state = self.get_image_state()
        done = (self.slot_index % self.episode_length == 0)
        if done:
            if self.slot_index + self.episode_length > self.trainset_info[1]: self.slot_index = 0
        self.bandwidth_buffer = [bandwidth, self.bandwidth_buffer[0], self.bandwidth_buffer[1],
                                 self.bandwidth_buffer[2]]
        self.state[2] = self.estimateBandwidth()
        return self.state, self.images_state, res[0], done, res[1], res[2], res[3]

    def estimateBandwidth(self):
        res = 0
        for w, b in zip(self.weight, self.bandwidth_buffer):
            res += w * b
        return res

    def getBandwidth(self):
        res = 0
        for i in range(int(self.slot_time / 0.1)):
            tmp = self.tracefile.readline()
            if tmp == "":
                self.tracefile = open("../traces/bandwidth/" + str(random.randint(0, 32)) + ".log", 'r')
                res += float(self.tracefile.readline())
            else:
                res += float(tmp)
        return res / int(self.slot_time / 0.1)

    def estimateReward(self, delay, accuracy, fps, v):

        reward = -self.a1 * delay + self.a2 * accuracy + self.a3 * (self.max_v / v) * (
                np.log(fps) / np.log(self.max_fps)) + self.bias
        # print(self.a1*delay,self.a2*accuracy,self.a3*(np.log(fps)/np.log(self.maxfps)))
        return reward, delay, accuracy, (self.max_v / v) * (np.log(fps) / np.log(self.max_fps))
        # return reward,delay,accuracy,(self.maxv/v)*(np.log(fps)/np.log(self.maxfps))

    def ReadRessultFile(self, resolution_index=None):
        res = []
        for count in range(5):
            t = self.train_data_file[count].readline()
            acc = self.train_data_file[count].readline()
            v = self.train_data_file[count].readline()
            if resolution_index == None:
                res.append((float(t), float(acc), float(v)))
            else:
                if count == resolution_index:
                    res = (float(t), float(acc), float(v))
            count += 1
        return res

    def get_image_state(self):
        return np.load(
            self.TRAIN_DATA_ROOT_PATH + self.train_path_name + "/feature_map/" + str(self.slot_index) + ".npy")


class env():
    def __init__(self, ImageBuffer, MotionVectorBuffer, original_fps=30):
        self.ImageBuffer = ImageBuffer
        self.MotionVectorBuffer = MotionVectorBuffer
        self.original_fps = original_fps
        self.frame_index = 0
        self.maxfps = original_fps
        self.minfps = 1
        self.re_range = [(480, 360), (720, 480), (1080, 720), (1600, 900), (1920, 1080)]
        self.N_S = 4  # last action,bandwidth,velocity
        self.A_S = (self.maxfps - self.minfps + 1) * len(self.re_range) * len(self.re_range)
        self.state = np.array([0, 0, 0, 0], np.double)
        self.historical_bandwidth = [0, 0, 0, 0, 0]

    def StateActiondim(self):
        return self.N_S, self.A_S

    def reset(self):
        self.state = np.array([20.0, 0, 100, 100], np.double)

        return self.state

    def step(self, actionindex):
        # fps=actionindex//(len(self.re_range)**2)+self.minfps
        # outer_resolution_index=actionindex%(len(self.re_range)**2)//len(self.re_range)
        # inter_resolution_index=actionindex%len(self.re_range)
        # self.state[0:3]=[fps,outer_resolution_index,inter_resolution_index]
        # print("take action:",fps,self.re_range[outer_resolution_index],self.re_range[inter_resolution_index])

        fps = actionindex // len(self.re_range) + self.minfps
        inter_resolution_index = actionindex % len(self.re_range)
        self.state[0:2] = [fps, inter_resolution_index]
        print("take action:", fps, self.re_range[inter_resolution_index])
        # select video
        video_stream = []
        index1 = 0
        index2 = 0
        for i in range(self.original_fps):
            while self.frame_index not in self.ImageBuffer.keys():
                pass
            if index1 >= index2:
                video_stream.append(self.ImageBuffer[self.frame_index])
                index2 += self.original_fps
            index1 += fps
            self.frame_index += 1

        video_bytes = encode(video_stream, self.re_range[inter_resolution_index])
        print("finish encoding")
        detecing_res, trans_latency, total_latency = upload_video(video_bytes)
        cur_bandwidth = len(video_bytes) / trans_latency
        self.historical_bandwidth.pop()
        self.historical_bandwidth.insert(0, cur_bandwidth)
        estimate_bandwidth = self.estimateBandwidth(self.historical_bandwidth)
        self.state[2] = estimate_bandwidth

        motion_index = self.frame_index - self.original_fps
        motion_vector_list = []
        print(len(self.MotionVectorBuffer), len(self.ImageBuffer))
        for i in range(self.original_fps):
            while motion_index not in self.MotionVectorBuffer.keys():
                pass
            motion_vector_list.append(self.MotionVectorBuffer[motion_index])
            motion_index += 1

        self.state[3] = self.estimateObjectVelocity(motion_vector_list)
        done = False
        return self.state, done

    def estimateBandwidth(self, previous_banwidth):
        weight = [0.5, 0.2, 0.15, 0.1, 0.05]
        res = 0
        for w, b in zip(weight, previous_banwidth):
            res = res + w * b
        return res

    def estimateObjectVelocity(self, motion_vector_list):
        v_list = []
        for motion_vectors in motion_vector_list:
            if motion_vectors != None:
                v = 0
                for mv in motion_vectors:
                    v += (abs(mv.x) + abs(mv.y))
                v /= len(motion_vectors)
                v_list.append(v)
        if len(v_list) == 0:
            return 0
        velocity = sum(v_list) / len(v_list)
        return velocity
