import cv2

def videoPlayer(frame_buffer,res_buffer,fps_buffer):
    print("start video player!")
    while True:
        fps=fps_buffer.get()
        time_per_frame=int(1000/fps)
        for i in range(fps):
            f=frame_buffer.get()
            r=res_buffer.get()
            for box in r:
                cv2.rectangle(f,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),0,2)
            cv2.putText(f,str(fps),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv2.imshow('frame',f)
            cv2.waitKey(time_per_frame)

    cap.release()
    cv2.destroyAllWindows()

