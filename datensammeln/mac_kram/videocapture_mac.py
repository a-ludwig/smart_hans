import enum
import numpy as np
import keyboard
import cv2
from datetime import datetime
from pip import main
import time
import threading
import os
import vlc
import sys
from os.path import basename, expanduser, isfile, join as joined



##macos tkinter stuff



if sys.version_info[0] < 3:
    import Tkinter as Tk
    from Tkinter import ttk
    from Tkinter.filedialog import askopenfilename
    from Tkinter.tkMessageBox import showerror
else:
    import tkinter as Tk
    from tkinter import ttk
    from tkinter.filedialog import askopenfilename
    from tkinter.messagebox import showerror

_isMacOS   = sys.platform.startswith('darwin')
_isWindows = sys.platform.startswith('win')
_isLinux   = sys.platform.startswith('linux')

class Window(Tk.Tk):
    def register(self, player):
        self.attributes("-fullscreen", True)
        id = self.winfo_id()
        print(id)

        if _isMacOS:
            player.set_nsobject(_GetNSView(id))
        elif _isLinux:
            player.set_xwindow(id)
        elif _isWindows:
            player.set_hwnd(id)

if _isMacOS:
    from ctypes import c_void_p, cdll
    # libtk = cdll.LoadLibrary(ctypes.util.find_library('tk'))
    # returns the tk library /usr/lib/libtk.dylib from macOS,
    # but we need the tkX.Y library bundled with Python 3+,
    # to match the version number of tkinter, _tkinter, etc.
    try:
        libtk = 'libtk%s.dylib' % (Tk.TkVersion,)
        prefix = getattr(sys, 'base_prefix', sys.prefix)
        libtk = joined(prefix, 'lib', libtk)
        dylib = cdll.LoadLibrary(libtk)
        # getNSView = dylib.TkMacOSXDrawableView is the
        # proper function to call, but that is non-public
        # (in Tk source file macosx/TkMacOSXSubwindows.c)
        # and dylib.TkMacOSXGetRootControl happens to call
        # dylib.TkMacOSXDrawableView and return the NSView
        _GetNSView = dylib.TkMacOSXGetRootControl
        # C signature: void *_GetNSView(void *drawable) to get
        # the Cocoa/Obj-C NSWindow.contentView attribute, the
        # drawable NSView object of the (drawable) NSWindow
        _GetNSView.restype = c_void_p
        _GetNSView.argtypes = c_void_p,
        del dylib
    except (NameError, OSError):  # image or symbol not found
        def _GetNSView(unused):
            return None
        libtk = "N/A"

    C_Key = "Command-"  # shortcut key modifier

else:  # *nix, Xwindows and Windows, UNTESTED

    libtk = "N/A"
    C_Key = "Control-"  # shortcut key modifier


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
    tap_num = 15
    duration = 5
    framerate = 30.0
    
    
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # had to be set for Linux
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2 )
    #Set highest possible resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(width)
    print(height)

    print("Please type acronym for: Geschlecht(m/w/d), Größe(k/n/g), Gesicht frei?(y/n)")
    kennung = input()

    num = int(input("Type in number you think of and press Enter to continue...: "))

    path = "videos/pause_" + str(tap_pause) + "/"
    
    if( not os.path.isdir(path)):
        os.mkdir(path)

    duration = tap_num * (tap_pause+1)
    ############
    #gui stuff for mac_os
    ############
    # tk_root = Tk.Tk()
    # #tk_root.attributes("-fullscreen", True) 
    # tk_window = Tk.Frame(tk_root)
    # tk_video = ttk.Frame(tk_window)
    # h = tk_video.winfo_id()
    # v = _GetNSView(h)

 
    #############
    # vlc stuff #
    #############

    #create instance
    vlc_instance = vlc.Instance('--no-video-title-show', '--fullscreen','--video-on-top', '--mouse-hide-timeout=0')
    
    #get audio_outputs
    auout_list =vlc.libvlc_audio_output_list_get(vlc_instance)
    print(auout_list)
    audioutputs = vlc_instance.audio_output_enumerate_devices()
    print(type(auout_list[1]))
    print (audioutputs)
    audio_out1 = audioutputs[1]['name']
    audio_out2 = audioutputs[1]['name']

    audioutput_device_list = vlc_instance.audio_output_device_list_get(audio_out1)
    print (audioutput_device_list)

    
    #create media_players
    player1 = vlc.MediaPlayer(vlc_instance)
    player2 = vlc.MediaPlayer(vlc_instance)

    # if _isMacOS :
    #     if v:        
    #         player1.set_nsobject(v)
    #     else :
    #         player1.set_ns_object(h)
   # player2.set_nsobject(tk_window.winfo_id()
    window = Window()
    window2 = Window()
    window.register(player1)
    window2.register(player2)

    

    # print(player1.audio_get_channel())
    # print(player2.audio_get_channel())
    # player1.audio_set_channel(1)
    # player2.audio_set_channel(2)
    # print(player1.audio_set_channel(1))
    # print(player2.audio_set_channel(2))
    # print(player2.audio_set_volume(0))
    # print(player1.audio_get_channel())
    # print(player2.audio_get_channel())
    #player1.set_fullscreen(True)
   # player2.set_fullscreen(True)
    #vlc.libvlc_audio_output_set(player1, audio_out1)
    #vlc.libvlc_audio_output_set(player2, audio_out2)
    # multithreading:
    print("Main    : before creating thread")



    thread_play_tap = threading.Thread(target=playTap, args=(num,tap_num, player2 ))
    thread_record = threading.Thread(target=recordVideo, args=(cap, duration, framerate, num, tap_pause, path, kennung))
    print("Main    : before running thread")
    
   # playIdle(player1)
     

    thread_record.start()
    thread_play_tap.start()
    window.mainloop() 
    
    
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

def playIdle(vlc_instance ):

    # if _isMacOS:
    #     vlc_instance.set_nsobject(h)

    playing = True
    media = vlc.Media("Static_664_1080.mp4")
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
            ######## have to be root for using keyboard on linux - big nope nope :( #########
            #if keyboard.is_pressed('q'): # if key 'q' is pressed 
            if debug: 
                #print('You Pressed A Key!')
                playing = False
            if vlc_instance.is_playing() == 0:
                break

def playTap(num, tap_num, vlc_instance):
    global curr_num, start
    
    media = vlc.Media("Horse_Tapping_Loop_664_1080.mp4")
    vlc_instance.set_media(media)
    vlc_instance.play()

    media = vlc.Media("Horse_Tapping_Loop_664_1080.mp4")
   
    
    for i in range(tap_num):
        vlc_instance.set_media(media)
        vlc_instance.play()

        time.sleep(0.2)
        while True:
            if vlc_instance.is_playing() == 0:
                break
        curr_num = i + 1
    media = vlc.Media("Static_664_1080.mp4")
    vlc_instance.set_media(media)
    vlc_instance.play()


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
        # cv2.putText(frame, str(i), [300,100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # cv2.putText(frame, "current number: " + str(curr_num), [400,100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        #cv2.imshow('frame', frame)
        #cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 2)

        if curr_num == num-1:
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
    out = cv2.VideoWriter(capturestring, fourcc, 30.0, (1080,  1920))

    for frame in frameNP:
        out.write(np.rot90(frame))
    out.release()    

if __name__ == "__main__":
    main()