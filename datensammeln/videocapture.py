


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
start = False
def main():
    
    tap_pause = 0.8
    duration = 10
    framerate = 30.0
    
    cap = cv2.VideoCapture(0)
    #Set highest possible resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    print(width)
    print(height)

    print("Please type acronym for: Geschlecht(m/w/d), Größe(k/n/g), Gesicht frei?(y/n)")
    kennung = input()

    num = int(input("Type in number you think of and press Enter to continue...: "))

    path = "videos/pause_" + str(tap_pause) + "/"
    
    if( not os.path.isdir(path)):
        os.mkdir(path)

    # multithreading:
    print("Main    : before creating thread")
    thread_play_tap = threading.Thread(target=playTap, args=(num,tap_pause))
    #thread_play_tap.isDaemon()
    thread_record = threading.Thread(target=recordVideo, args=(cap, duration, framerate, num, tap_pause, path, kennung))
    print("Main    : before running thread")
    thread_play_tap.start()
    thread_record.start()
    print("Main    : wait for the thread to finish")
    #thread_play_tap.join()
    print("Main    : all done")
    
    
    #recordVideo(cap, capturestring, duration,  framerate)
  #  from datetime import datetime

def playTap(num, tap_pause):
    global curr_num, start

    # def im_show(img, name, time):
    #  cv2.namedWindow(name)
    #  cv2.moveWindow(name, 900,-900)
    #  cv2.imshow(name, img)
    #  cv2.waitKey(time)
    # return

    

    # while not start:

    #     try:
    #         vs = cv2.VideoCapture("HANS_Repo\datensammeln\Looking_Around.mp4")
    #     except:
    #         print("Video file not found")
    #     ret, img = vs.read()

    #     try:
    #             #cv2.startWindowThread()
    #             img = imutils.resize(img, width=480)
                
    #             cv2.imshow("Horse Tapping", img)
    #             cv2.setWindowProperty("Horse Tapping", cv2.WND_PROP_TOPMOST, 1)
                
    #             cv2.waitKey(1)
    #     except:
    #         break
        


    for i in range(num+3):
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

def createFilename(infos):
    filename = "smart_hans"
    for info in infos:
        filename = filename + "_" + str(info)
    filename = filename + ".avi"
    return filename


def recordVideo(cap, duration, framerate, num, tap_pause, path, kennung):

    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")
    
    #Define amount of Frames to Record
    frames_to_record = int(duration*framerate)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frameNP = np.zeros(shape=(frames_to_record,1080,1920,3),dtype=np.uint8)
    targetFrame = []
    
    

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

        if curr_num == num or curr_num == num-1:
            targetFrame.append(str(i))

        if cv2.waitKey(1) == ord('q'):
            break
        i+=1
        print(i)
       
        if i >= frames_to_record:
            break
    # Release everything if job is finished
    print(frameNP[0].shape)
    cap.release()
    anmerkungen = input("Anmerkungen?: ")

    infos = [date_time, num, targetFrame[0] + "-" + targetFrame[-1], tap_pause, kennung, anmerkungen]

    filename = createFilename(infos)
    capturestring = (path + filename)


    print(capturestring)
    out = cv2.VideoWriter(capturestring, fourcc, 30.0, (1920,  1080))

    for frame in frameNP:
        out.write(frame)
    out.release()    

if __name__ == "__main__":
    main()