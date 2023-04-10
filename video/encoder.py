import cv2
import os
def encode(VideoStream,inter_resolution,outer_resolution=0):
    fourcc=cv2.VideoWriter_fourcc(*'X264')
    videoWriter = cv2.VideoWriter('H264_TEMP.264',fourcc, len(VideoStream), inter_resolution)
    for f in VideoStream:
        videoWriter.write(f)
    videoWriter.release()

    h264_file=open('H264_TEMP.264','rb')
    h264_stream=h264_file.read()
    h264_file.close()
    os.remove('H264_TEMP.264')

    return h264_stream
