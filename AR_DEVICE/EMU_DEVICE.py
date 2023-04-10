import cv2

def start_emuAR_device(ImageBuffer):
    #video_path="C:/Users/user/Desktop/others/Motorway_Traffic.mp4"
    video_path="C:/Users/user/Desktop/others/exp.avi"
    cap = cv2.VideoCapture(video_path)
    frame_index=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            ImageBuffer[frame_index]=frame
            frame_index+=1
