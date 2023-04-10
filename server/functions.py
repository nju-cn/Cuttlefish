#ONLY ".avi" SUPPORTED
import cv2
import os

def read_video(video_path):
    video_capture=cv2.VideoCapture(video_path)
    fps=int(video_capture.get(cv2.CAP_PROP_FPS))
    size=(int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    totalframe = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return video_capture,fps,size,totalframe

def division(video_path,dst_dir_path,slot_time):
    cap,fps,size,totalframe = read_video(video_path)
    frame_in_slot=slot_time*fps
    slotnum=int(totalframe/frame_in_slot)
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)
    print("start divide video...")
    for i in range(slotnum):
        videowriter = cv2.VideoWriter(dst_dir_path+"/"+str(i)+".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
        index=0
        while index<frame_in_slot:
            success, img=cap.read()
            if success:
                videowriter.write(img)
            index+=1
    print("finished!")

def resize_video(video_path,dst_path,target_size):
    cap,fps,size,_ = read_video(video_path)
    videowriter = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, target_size)
    success,img=cap.read()
    while success:
        try:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            videowriter.write(img)
        except:
            print("errors in RESIZE process")
        success, img = cap.read()

def reFps_video(video_path,dst_path,target_fps):
    cap,fps,size,total = read_video(video_path)
    if target_fps>=fps:
        target_fps=fps
    videowriter = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), target_fps, size)
    index1=0
    index2=0
    success,img=cap.read()
    while success:
        if index1>=index2:
            videowriter.write(img)
            index2+=fps
        index1+=target_fps
        success, img = cap.read()


