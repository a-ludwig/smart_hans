


import enum
import numpy as np
import imutils
#import keyboard
import cv2
from datetime import datetime
from pip import main
import time
import threading
import os

curr_num = 0

def main():
    
    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")
    capturestring = ("videos/smart_hans_" + date_time +".avi")
    tap_pause = 1.2
    duration = 2
    framerate = 30.0
    print(capturestring)
    cap = cv2.VideoCapture(0)
    #Set highest possible resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    print(width)
    print(height)

    num = int(input("Type in number you think of and press Enter to continue...: "))

    path = "videos/pause_" + str(tap_pause) + "/"
    filename = "smart_hans_" + date_time + "_target_number_"+ str(num) + ".avi"

    if( not os.path.isdir(path) ):
        os.mkdir(path)

    capturestring = (path + filename)

    # multithreading:
    print("Main    : before creating thread")
    thread_play_tap = threading.Thread(target=playTap, args=(num,tap_pause))
    #thread_play_tap.isDaemon()
    thread_record = threading.Thread(target=recordVideo, args=(cap, capturestring, duration,framerate))
    print("Main    : before running thread")
    thread_play_tap.start()
    thread_record.start()
    print("Main    : wait for the thread to finish")
    #thread_play_tap.join()
    print("Main    : all done")
    
    
    #recordVideo(cap, capturestring, duration,  framerate)
  #  from datetime import datetime

def playTap(num, tap_pause):
    global curr_num
    
    for i in range(num):
        time.sleep(tap_pause)
        try:
            vs = cv2.VideoCapture("HANS_Repo\datensammeln\Horse_Tapping_One_Tap.mp4")
        except:
            print("Video file not found")
        while True:
            ret, img = vs.read()

            try:
                #cv2.startWindowThread()
                img = imutils.resize(img, width=480)
                
                cv2.imshow("Horse Tapping", img)
                cv2.setWindowProperty("Horse Tapping", cv2.WND_PROP_TOPMOST, 1)
                
                cv2.waitKey(1)
            except:
                break
        curr_num = i + 1
        
    vs.release()
    #cv2.destroyAllWindows();


def recordVideo(cap, capturestring, duration, framerate):
    
    #Define amount of Frames to Record
    frames_to_record = int(duration*framerate)

# Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frameNP = np.zeros(shape=(frames_to_record,1080,1920,3),dtype=np.uint8)
    out = cv2.VideoWriter(capturestring, fourcc, 30.0, (1920,  1080))
    
    
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        #print(frame.shape)
        
       
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
       # frame = cv2.flip(frame, 1)
        # write the flipped frame
        frameNP[i]=frame
        cv2.putText(frame, str(i), [300,100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame, "current number: " + str(curr_num), [400,100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('frame', frame)
        cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 2)
        
        if cv2.waitKey(1) == ord('q'):
            break
        i+=1
        print(i)
       
        if i >= frames_to_record:
            break
    # Release everything if job is finished
    print(frameNP[0].shape)
    cap.release()

    for frame in frameNP:
        out.write(frame)
    out.release()    

if __name__ == "__main__":
    main()