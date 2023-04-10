import cv2
import os
def create_video_slot(video_name,slot_time=1):
    cap=cv2.VideoCapture(video_name)
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    if not os.path.exists("video"):
        os.mkdir("video")
    target_path="video/"+video_name+"_slot/"
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    success,frame=cap.read()
    index=0
    video_writer=cv2.VideoWriter(target_path+"0.avi",cv2.VideoWriter_fourcc('M','J','P','G'),fps,(1920,1080))
    while success:
        index+=1
        video_writer.write(frame)
        if index%fps==0:
            video_writer=cv2.VideoWriter(target_path+str(index//fps)+".avi",cv2.VideoWriter_fourcc('M','J','P','G'),fps,(1920,1080))
        success,frame=cap.read()

create_video_slot("exp.avi")
        
