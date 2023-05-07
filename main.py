import cv2
import numpy as np
import math
import time
from datetime import datetime as archi
import os



from imutils.video import FPS

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


from Machine_learning.datenverarbeitung.dataloader import dataloader
from utils.horse import horse
from utils.helper import * 
import pandas as pd
import numpy as np


from tsai.all import *
from threading import Thread

rot_angle = 90 #270
debug = True
found_face = False
stop_idle = False
stop_tap = False
curr_num = 1
df = pd.DataFrame()


##threshold for distance detection
DIST_THRESH = 35
##waiting time for Hansi to Start in Secs
WAITING_TIME = 5
PI_RESPONSE_TIME = 7

## Socket connection:
# Define the IP address and port of the Raspberry Pi
IP_ADDRESS = '192.168.1.195'  # Replace with the IP address of your Raspberry Pi
PORT = 1234

def main():
    
    nr_taps = 1
    tap_size = 30
    window_size = tap_size * nr_taps
    move_by = -12
    prediction_threshold = 0.571
    counter = 0 
    cycle_size = 23
    n = 2 ### cycle/n for modulo

    

    min = 0
    max = 0
    #load tsai model
    predictor = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    #load hp model
    face_model, landmark_model = init_hp_model()

    hansi = horse(max_taps=12) ##hansi is our magnificent horse 

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    ret, img = cap.read()
    rot_M, cam_M = get_camera_matrixes(img, rot_angle)

    

    dl = dataloader(scenario = 3, nr_taps = nr_taps, tap_size = tap_size, move_window_by = move_by, feature_list = ['chin_y'] )
    num_params = len(dl.column_dict)-1
    
    timer_in_sec, last_t, data = init_params(num_params)

    font = cv2.FONT_HERSHEY_SIMPLEX 
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    ## Face distance
    dist = 0
    curr_win_size = 0
    # dl.window_size = 28
    video_stream_widget = VideoStreamWidget()
    fps_arr = np.array([])
    rand_int = random.randint(3, 10)
    while True:
        try:
            #video_stream_widget.show_frame()
            img = video_stream_widget.frame
            fps = FPS().start()

            img = cv2.warpAffine(img, rot_M, (img.shape[1], img.shape[0]))## rot image
            hansi.queue()## keep hansi alive

            faces, face_found = find_face(img, face_model)
            
            if not face_found:
                hansi.switch = "idle"
                dist = 0 ##resets the timer
                hansi.curr_win_size = 0
                fps_arr = np.array([])
                rand_int = random.randint(3, 10)
            else:
                img, image_points, all_points = estimate_head_pose(img, model_points, cam_M, face_model, landmark_model, faces)

                dist = get_face_dist(image_points)
                #print(f"hansi curr tap: {hansi.curr_tap}")
                if hansi.switch == "tapping" and hansi.curr_tap >= 2:
                    data.append(all_points)
                    hansi.curr_win_size += 1
                    
                    #######
                    #Make prediction  n times per cycle(tap)
                    #######
                    #print(hansi.curr_tap)
                    if (hansi.curr_win_size % (cycle_size+move_by) == 0)and hansi.curr_tap >= 3: #delim % (cycle_size/n) == 0 and hansi.curr_tap >= 3: 

                        print(f"im predicting at calc_tap:{hansi.curr_tap}")
                        #print(delim)
                        #print("***predicting***")
                        window_scaled, min, max = list_to_norm_win(dl, data, min, max)
                        predicted_class = make_pred(window_scaled, predictor, threshold=prediction_threshold, class_to_look_at=1)
                        #print(f"predicted class: {predicted_class}")
                        if predicted_class == 1:
                            hansi.switch = "end_tap"
                            hansi.save = True
                            hansi.pred_tap  = hansi.curr_tap
                            

                            #### reset for new participant
                
                    if hansi.switch == "announce_end":
                        data.append(all_points)

            timer_in_sec, last_t = wait_for_face(timer_in_sec, last_t, dist, DIST_THRESH)

            if int(timer_in_sec) == WAITING_TIME and hansi.switch == "idle":
                hansi.switch = "start_tap"
            elif timer_in_sec < WAITING_TIME and hansi.switch == "tapping": 
                hansi.switch = "end_tap"

            if hansi.switch == "reset_idle":
                if hansi.save == True:
                    df2 = pd.DataFrame(data)
                    now = archi.now()
                    date_time = now.strftime("%m%d%Y_%H%M%S")
                    fps_avg = int(np.mean(fps_arr))

                    feedback = send_rec_feedback(IP_ADDRESS, PORT, hansi, PI_RESPONSE_TIME)

                    filename = f"installation_export/inst_exp_{date_time}_tap_{hansi.pred_tap}_fdb_{feedback}_avgFPS_{fps_avg}.csv"
                    df2.to_csv(filename)

                    ## new export with labeled data:
                    ## add "WindowOfInterest_tapnumber" to end of filename
                    #filename = f"installation_export/inst_exp_{date_time}_{hansi.target_frame[0]}-{hansi.target_frame[-1]}_.csv"
                    #df2.to_csv(filename)

                    #df2.to_csv(f"installation_export/inst_exp_{date_time}.csv")

                    hansi.save = False
                timer_in_sec, last_t, data = init_params(num_params)

            
            # update the FPS counter
            fps.update()
            fps.stop()
            
            fps_arr = np.append(fps_arr, int(fps.fps()))


            color = (0,255,0) if stop_idle else (255,0,0)

            # cv2.putText(img, str(int(timer_in_sec)), [100,100], font, 2, color, 3)
            # cv2.putText(img, str(dist), [180,100], font, 2, color, 3)
            # cv2.putText(img, str(int(fps.fps())), [100,200], font, 2, (0,0,255), 3)

            counter = counter +1
            cv2.imshow('img', img)
            cv2.imwrite("C:/Users/adi/Documents/hansi_dokuplakat/single_Frames/bepunktet_neu_%d.jpg" % counter, img) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except AttributeError as e:
            print(f"An AttributeError occurred: {e}")
            # you can also choose to log the error, raise a different exception, or take other actions as needed
            pass
    cv2.destroyAllWindows()
    cap.release()

def init_params(num_params):
    data = []
    timer_in_sec = 0 # our variable for *absolute* time measurement
    last_t = 0 # cache var
    return timer_in_sec, last_t, data

def make_pred( window_scaled,predictor, threshold, class_to_look_at):
    
    '''Takes in dataloader object, a windowed dataframe, a tsai predictor and a custom threshold according to the problem.
        Returns None (default) if no class with probability above threshold is received.'''
    class_predicted = None
    
    X = window_scaled
    
    X = np.array([X])
    X = np.array([X])
    
    print(X)
    ##Inference on Window
    #
    probabilities_class, _, predicted_class = predictor.get_X_preds(X)
    
    predictor_probas_np = probabilities_class.numpy()[0]
    print(probabilities_class)
    class_predicted = None
    temp = 0
    # for i, elem in enumerate(predictor_probas_np):
    #                     if elem >= threshold and elem >= temp:
    #                         class_predicted = i
    #                         temp = elem
                            #print(elem)
    if predictor_probas_np[class_to_look_at] > threshold:
        class_predicted = class_to_look_at

    ##return: default: None, otherwise Class
    return class_predicted

def list_to_norm_win(dl, data, min, max):
    ##
    delim = -dl.window_size #- dl.move_window_by if dl.move_window_by < 0 else -dl.window_size
    window_arr = np.array(data[delim:])
    
    ##work around, actually only one feature
    for elem in dl.feature_list:
        index = dl.column_dict[elem]
        window_arr_for_norm = window_arr[:,index]

    ##########
    ###Normalization right here instead of dl
    temp_max = np.amax(window_arr_for_norm)
    temp_min = np.amin(window_arr_for_norm)
    divisor = temp_max - temp_min#max-min
    sub_arr = np.array([temp_min] * len(window_arr_for_norm))

    if divisor == 0 :
       
        print("dropping frame")
        divisor = 0.5
    
    window_scaled = (window_arr_for_norm - temp_min)/ divisor
    
    return window_scaled, min, max


def wait_for_face(timer_in_sec, last_t, dist, thresh):
    #global found_face, stop_idle
    thresh = thresh

    # detection loop
    now = time.time() # in seconds 
    dt = now - last_t # diff time
    last_t = now      # cache

    # face close enough?
    if dist > thresh:
        timer_in_sec += dt  # sum up
        #print(timer_in_sec)
    else:
        timer_in_sec = 0    # reset
    return timer_in_sec, last_t
    
if __name__ == "__main__":
    main()