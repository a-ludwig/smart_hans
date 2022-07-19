


import enum
import numpy as np
import imutils
import keyboard
import cv2
from datetime import datetime
from pip import main
import time
import threading
import os
from ffpyplayer.player import MediaPlayer
import vlc 


debug = True

def extract_image_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    success = True
    while success:
        success, image = cap.read()
        if image is None:
            continue
        yield image
    cap.release()

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

    duration = num+4 * (tap_pause+1)

    #############
    # vlc stuff #
    #############

    #create instance
    vlc_inst = vlc.Instance('--no-video-title-show', '--fullscreen','--video-on-top', '--mouse-hide-timeout=0')
    #create media_player
    vlc_inst = vlc.MediaPlayer(vlc_inst)
    vlc_inst.set_fullscreen(True)
    
    # multithreading:
    print("Main    : before creating thread")

    thread_play_tap = threading.Thread(target=playTap, args=(num,tap_pause, vlc_inst))
    thread_record = threading.Thread(target=recordVideo, args=(cap, duration, framerate, num, tap_pause, path, kennung))
    print("Main    : before running thread")

    playIdle(vlc_inst)

    thread_record.start()
    thread_play_tap.start()
    
    print("Main    : wait for the thread to finish")
    print("Main    : before running thread")
    print("Main    : all done")
    

# def im_show(img, name, time):
    #  cv2.namedWindow(name)
    #  cv2.moveWindow(name, 900,-900)
    #  cv2.imshow(name, img)
    #  cv2.waitKey(time)

    # return

# Currently not used

def playIdle(vlc_instance):
    playing = True
    media = vlc.Media("Looking_Around.mp4")
    vlc_instance.set_media(media)
    vlc_instance.play()

    while True:
        if vlc_instance.is_playing() == 0:
            break

    while playing:
        vlc_instance.set_media(media)
        vlc_instance.play()

        time.sleep(0.2)
        while True:
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print('You Pressed A Key!')
                playing = False
            if vlc_instance.is_playing() == 0:
                break

def playTap(num, tap_pause, vlc_instance):
    global curr_num, start
    extra_taps = 3
    
    media = vlc.Media("Horse_Tapping_One_Tap.mp4")
    
    for i in range(num+extra_taps):
        vlc_instance.set_media(media)
        vlc_instance.play()

        time.sleep(0.2)
        while True:
            if vlc_instance.is_playing() == 0:
                break
        curr_num = i + 1
    playIdle(vlc_instance)


def createFilename(infos):
    filename = "smart_hans"
    for info in infos:
        filename = filename + "_" + str(info)
    filename = filename + ".avi"
    return filename


def recordVideo(cap, duration, framerate, num, tap_pause, path, kennung):
    global curr_num
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

        frameNP[i]=frame
        cv2.putText(frame, str(i), [300,100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame, "current number: " + str(curr_num), [400,100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        #cv2.imshow('frame', frame)
        #cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 2)

        if curr_num == num or curr_num == num-1:
            targetFrame.append(str(i))

        if cv2.waitKey(1) == ord('q'):
            break
        i+=1
       
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