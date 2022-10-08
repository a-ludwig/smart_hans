from cProfile import label
from curses import window
from glob import glob
from lib2to3.pgen2.pgen import DFAState
from turtle import right
import cv2
import numpy as np
import math
import threading
import time
import keyboard
import os
import sys

from headpose_opencv.face_detector import get_face_detector, find_faces
from headpose_opencv.face_landmarks import get_landmark_model, detect_marks
from Machine_learning.datenverarbeitung.dataloader import dataloader
import pandas as pd
import numpy as np
import vlc

from tsai.all import *

rot_angle = 0
debug = True
found_face = False
stop_idle = False
stop_tap = False
curr_num = 1
df = pd.DataFrame()


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    """
    Draw a 3D anotation box on the face for head pose estimation

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    rear_size : int, optional
        Size of rear box. The default is 300.
    rear_depth : int, optional
        The default is 0.
    front_size : int, optional
        Size of front box. The default is 500.
    front_depth : int, optional
        Front depth. The default is 400.
    color : tuple, optional
        The color with which to draw annotation box. The default is (255, 255, 0).
    line_width : int, optional
        line width of lines drawn. The default is 2.

    Returns
    -------
    None.

    """
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways    

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)


def main():
    
    tap_pause = 0.8
    tap_num = 15
    duration = 5
    framerate = 30.0
    time_to_activate = 5
    
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #Set highest possible resolution
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )

    duration = tap_num * (tap_pause+1)
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
    thread_play_tap = threading.Thread(target=playTap, args=(tap_num, vlc_inst))
    #thread_record = threading.Thread(target=recordVideo, args=(cap, duration, framerate, num, tap_pause, path, kennung))
    thread_estimate_head_pose = threading.Thread(target= estimate_head_pose, args=())
    #thread_check_face = threading.Thread(target= check_face, args=())
    #thread_play_idle = threading.Thread(target= playIdle, args=(vlc_inst, duration))
    thread_vlc = threading.Thread(target = vlc_thread, args=(vlc_inst, tap_num))
    print("Main    : before running thread")
    #thread_estimate_head_pose.start()
    #thread_vlc.start()

    estimate_head_pose()


    #time.sleep(2)
    #thread_check_face.start()
    #thread_record.start()
    #thread_play_tap.start()
    
    print("Main    : wait for the thread to finish")
    print("Main    : before running thread")
    print("Main    : all done")

def vlc_thread(vlc_inst, tap_num):
    playIdle(vlc_inst, tap_num)
    playTap(tap_num, vlc_inst)



def wait_for_face(timer, last_t, dist, time_sec):
    global found_face, stop_idle
    time_to_activate = time_sec
    thresh = 40

    # detection loop
    now = time.time() # in seconds 
    dt = now - last_t # diff time
    last_t = now      # cache

    # try to detect person

    if dist > thresh:
        timer += dt  # sum up
    else:
        timer = 0    # reset
        stop_idle = False

    if timer > time_to_activate:
        stop_idle = True
    return timer, last_t
    


def estimate_head_pose():
    global found_face, stop_idle, curr_num
    #load tsai model
    predictor = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, img = cap.read()
    size = img.shape
    center = (size[1]/2, size[0]/2)
    M = cv2.getRotationMatrix2D(center, rot_angle, scale = 1)
    img = cv2.warpAffine(img, M, (size[1], size[0]))
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

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

    data = []
    stumpM = 30
    timer = 0 # our variable for *absolute* time measurement
    last_t = 0 # cache var
    nr_taps = 1
    tap_size = 40
    window_size = tap_size * nr_taps
    
    move_by = 0
    
    dl = dataloader(scenario = 3, nr_taps = nr_taps, move_window_by = move_by, feature_list = ['nosetip_y'] )
    num_params = len(dl.column_dict)-1
    dataset_np = np.empty((num_params))

    while True:
        ret, img = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if ret == True:
            #rot image
            img = cv2.warpAffine(img, M, (size[1], size[0]))
            faces = find_faces(img, face_model)
            for face in faces:
                #found_face = True
                marks = detect_marks(img, landmark_model, face)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array([
                                        marks[30],     # Nose tip
                                        marks[8],     # Chin
                                        marks[36],     # Left eye left corner
                                        marks[45],     # Right eye right corne
                                        marks[48],     # Left Mouth corner
                                        marks[54]      # Right mouth corner
                                    ], dtype="double")
                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
                
                
                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose
                
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                
                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                
                
                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                cv2.line(img, p1, p2, (0, 255, 255), 2)
                
                cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # for (x, y) in marks:
                #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90
                    
                try:
                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))
                except:
                    ang2 = 90
                

                dist = get_face_dist(image_points)
                
                timer, last_t = wait_for_face(timer, last_t, dist, 5)

                color = (0,255,0) if stop_idle else (255,0,0)
                
                cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
                cv2.putText(img, str(int(timer)), [100,100], font, 2, color, 3)
                cv2.putText(img, str(dist), [180,100], font, 2, color, 3)
                rmat, jac = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                pitch = np.arctan2(Qx[2][1], Qx[2][2])
                roll = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
                yaw = np.arctan2(Qz[0][0], Qz[1][0])

               # tap.isPlaying:
                dataset_np = np.vstack ([dataset_np, np.array([
                    image_points[0][0],
                    image_points[0][1],
                    image_points[1][0],
                    image_points[1][1],
                    image_points[2][0],
                    image_points[2][1],
                    image_points[3][0],
                    image_points[3][1],
                    image_points[4][0],
                    image_points[4][1],
                    image_points[5][0],
                    image_points[5][1],
                    p2[0],
                    p2[1],
                    x1[0],
                    x1[1],
                    x2[0], 
                    x2[1],
                    pitch,
                    roll,
                    yaw,
                    ])])

               # print(dataset_np)

            if dataset_np.shape[0] % window_size == 0 :
                ##
                delim = -window_size + move_by
                window_arr = dataset_np[delim:]

                ##np to df normalize and predict
                #

                feature_arr_list = []
                

                for elem in dl.feature_list:
                    index = dl.column_dict[elem]
                    feature_arr_list.append(window_arr[:,index])
                
                dl.univariate = False
                dl.col_names = dl.get_col_names(dl.window_size)
                np_for_norm = np.array([dl.col_names])

                for j, elem in enumerate(feature_arr_list):
                    labeled_window = np.append(np.array([1, j+1]), elem)
                    labeled_window = np.append(labeled_window, np.array(['target', 'filename']))
                    np_for_norm = dl.stack_dataset(np_for_norm, labeled_window)

                dataset_df  = pd.DataFrame(np_for_norm[1:].tolist(), columns=dl.col_names, dtype="float64")
                df_normalized = dl.normalize_df(dataset_df).iloc[ :, 2:-2]

                X = df_normalized.to_numpy()
                
                X = np.array([X])
                test_probas, test_targets, test_preds = predictor.get_X_preds(X, with_decoded=True)
                print(test_probas, test_targets, test_preds)
                print(X)

               # to_df_and_window(image_points = data ,tap_num = curr_num, window_size = window_size, move_by = move_by)
            ##############
            #cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

def get_face_dist(image_points):
    chin_y = image_points[1][1]
    right_mouth_corner_y = image_points[5][1]

    dist = abs(right_mouth_corner_y - chin_y)
    return dist

def playIdle(vlc_instance, duration):
    global stop_idle
    media = vlc.Media("datensammeln/tap_loop_start0900-1050.mp4")
    vlc_instance.set_media(media)
    vlc_instance.play()
    print("start idle")
    print(stop_idle)

    while True:
        if vlc_instance.is_playing() == 0:
            break
    
    while stop_idle == False:
        vlc_instance.set_media(media)
        vlc_instance.play()

        time.sleep(0.2)
        while True:
            if vlc_instance.is_playing() == 0:
                break


def playTap(tap_num, vlc_instance):
    global curr_num, stop_tap
    
    media = vlc.Media("datensammeln/tap_loop_start0001-0059.mp4")
    
    vlc_instance.set_media(media)
    vlc_instance.play()
    time.sleep(0.2)
    while True:
        if vlc_instance.is_playing() == 0:
            break

    curr_num = 1

    media = vlc.Media("datensammeln/tap_loop_start0060-0088.mp4")
    
    for i in range(tap_num-1):
        vlc_instance.set_media(media)
        vlc_instance.play()

        time.sleep(0.2)
        while True:
            if vlc_instance.is_playing() == 0:
                break
        curr_num = i + 1
        if stop_tap:
            break
    
    
    media = vlc.Media("datensammeln/tap_loop_start0118-0139.mp4")
    vlc_instance.set_media(media)
    vlc_instance.play()
    time.sleep(0.2)
    while True:
        if vlc_instance.is_playing() == 0:
            break
    #set curr_num to stop recording
    curr_num = -1

    media = vlc.Media("datensammeln/tap_loop_start0900-1050.mp4")
    vlc_instance.set_media(media)
    vlc_instance.play()
def to_df_and_window(image_points,tap_num, window_size, move_by):
    global df 
    ##append image_points to continouus data stream
    image_points 
    ##data in window füllen -> df nach if erzeugen -> data leeren 
    if len() >= tap_num * window_size + move_by:
        window = get_window_from_df(df,tap_num,window_size,move_by)

        print (window)
    
def get_window_from_df(df, tap_num, window_size, move_by):
    ##get window from current df
    #define delimeter 
    start_del = tap_num  * window_size + move_by
    end_del = tap_num * window_size + move_by

    #fill temp arr and append to target_class_array
    window_arr = df[start_del : end_del]
    return window_arr

if __name__ == "__main__":
    main()