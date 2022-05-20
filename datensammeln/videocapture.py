


import enum
import numpy as np
import cv2
from datetime import datetime
from pip import main

def main():
    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H:%M:%S")
    capturestring = ("videos/smart_hans_" + date_time +".avi")
    print(capturestring)
    cap = cv2.VideoCapture(0)
    #Set highest possible resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    print(width)
    print(height)
    recordVideo(cap, capturestring)
  #  from datetime import datetime

    

def recordVideo(cap, capturestring):

# Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameNP = np.zeros(shape=(150,720,1280,3),dtype=np.uint8)
    out = cv2.VideoWriter(capturestring, fourcc, 30.0, (1280,  720))
    
   

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        #print(frame.shape)
        
       
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.flip(frame, 0)
        # write the flipped frame
        frameNP[i]=frame
        cv2.putText(frame, str(i), [300,100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        i+=1
        print(i)
       
        if i >= 150:
            break
    # Release everything if job is finished
    print(frameNP[0])
    print(frameNP[0].shape)
    cap.release()

    for frame in frameNP:
        print(frame.shape)
        out.write(frame)
    out.release()    

if __name__ == "__main__":
    main()