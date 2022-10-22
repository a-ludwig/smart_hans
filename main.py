import cv2
import numpy as np
import math
import time
import os


from imutils.video import FPS

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from headpose_opencv.face_detector import get_face_detector, find_faces
from headpose_opencv.face_landmarks import get_landmark_model, detect_marks
from Machine_learning.datenverarbeitung.dataloader import dataloader
from utils.player import media_player
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
    
    nr_taps = 1
    tap_size = 35
    window_size = tap_size * nr_taps
    move_by = 0

    #load tsai model
    predictor = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    #load hp model
    face_model = get_face_detector()
    landmark_model = get_landmark_model()

    player = media_player()

    

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, img = cap.read()
    rot_M, cam_M = get_camera_matrixes(img, rot_angle)

    

    dl = dataloader(scenario = 3, nr_taps = nr_taps, tap_size = tap_size, move_window_by = move_by, feature_list = ['right_eye_corner_x'] )
    num_params = len(dl.column_dict)-1
    
    dataset_np, timer_in_sec, last_t = init_params(num_params)

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

    ## Gameloop - 30FPS repeating execution loop
    while True:
        ret, img = cap.read()
        fps = FPS().start()
        
        if ret == True:
            #fps = cap.get(cv2.CAP_PROP_FPS)
            #rot image
            img = cv2.warpAffine(img, rot_M, (img.shape[1], img.shape[0]))
            #dist = 0

            #########
            #datathread now handling vlc also -> no thread

            player.queue()

            faces, face_found = find_face(img, face_model)
            
            if face_found:
                img, image_points, all_points_np = estimate_head_pose(img, model_points, cam_M, face_model, landmark_model, faces)

                dist = get_face_dist(image_points)

                if player.switch == "tapping" and player.curr_tap >= 1:
                    dataset_np = np.vstack ([dataset_np, all_points_np])
                    curr_win_size += 1
                    
                    #print(dataset_np.shape[0])

                    if  dataset_np.shape[0] % 28 == 0 and player.curr_tap >= 3: #modulo von dataset_len % dl.window_size + abs(dl.moveby)
                        #curr_win_size = 0
                        print(dataset_np.shape[0])
                        print(f"im predicting at tap:{player.curr_tap}")
                        predicted_class = make_pred(dl, dataset_np, predictor, threshold=0.7)
                        print(f"predicted class: {predicted_class}")
                        if predicted_class == 1:
                            player.switch = "end_tap"

            else:
                player.switch = "idle"
                dist = 0
                curr_win_size = 0
            

            timer_in_sec, last_t = wait_for_face(timer_in_sec, last_t, dist)

            if int(timer_in_sec) == 5 and player.switch == "idle":
                player.switch = "start_tap"
            elif timer_in_sec < 5 and player.switch == "tapping": 
                player.switch = "end_tap"

            
            
            # update the FPS counter
            fps.update()
            fps.stop()


            color = (0,255,0) if stop_idle else (255,0,0)

            cv2.putText(img, str(int(timer_in_sec)), [100,100], font, 2, color, 3)
            cv2.putText(img, str(dist), [180,100], font, 2, color, 3)
            cv2.putText(img, str(int(fps.fps())), [100,200], font, 2, (0,0,255), 3)

            #### reset for new participant
            if player.switch == "end_tap":
                dataset_np, timer_in_sec, last_t = init_params(num_params)
                #print("restart")
                #### dave dataset if you want


           # to_df_and_window(image_points = data ,tap_num = curr_num, window_size = window_size, move_by = move_by)
            ##############
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    return

def init_params(num_params):
    dataset_np = np.empty((num_params))
    timer_in_sec = 0 # our variable for *absolute* time measurement
    last_t = 0 # cache var
    return dataset_np, timer_in_sec, last_t

def make_pred(dl, dataset_np, predictor, threshold):
    
    '''Takes in dataloader object, a windowed dataframe, a tsai predictor and a custom threshold according to the problem.
        Returns None (default) if no class with probability above threshold is received.'''
    class_predicted = None
    
    ##
    delim = -dl.window_size + dl.move_window_by
    window_arr = dataset_np[delim:]
    #print(window_arr)

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
    df_normalized = dl.normalize_df_by_window(dataset_df).iloc[ :, 2:-2]

    X = df_normalized.to_numpy()
    
    X = np.array([X])
    
    #print(X)
    ##Inference on Window
    #
    probabilities_class, _, predicted_class = predictor.get_X_preds(X, with_decoded=True)
    
    predictor_probas_np = probabilities_class.numpy()[0]
    
    class_predicted = None
    temp = 0
    for i, elem in enumerate(predictor_probas_np):
                        if elem >= threshold and elem >= temp:
                            class_predicted = i
                            temp = elem
                            #print(elem)

    ##return: default: None, otherwise Class
    return class_predicted

def get_camera_matrixes(img, rot_angle,):
    size = img.shape
    center = (size[1]/2, size[0]/2)
    M = cv2.getRotationMatrix2D(center, rot_angle, scale = 1)
    img = cv2.warpAffine(img, M, (size[1], size[0]))

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    return M, camera_matrix


def wait_for_face(timer_in_sec, last_t, dist):
    #global found_face, stop_idle
    thresh = 35

    # detection loop
    now = time.time() # in seconds 
    dt = now - last_t # diff time
    last_t = now      # cache

    # try to detect person

    if dist > thresh:
        timer_in_sec += dt  # sum up
        #print(timer_in_sec)
    else:
        timer_in_sec = 0    # reset
    return timer_in_sec, last_t
    
def find_face(img, face_model):
    faces = find_faces(img, face_model)
    if len(faces) == 0:
        found = False
    else:
        found = True
    return faces, found

def estimate_head_pose(img, model_points, camera_matrix, face_model, landmark_model, faces):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #faces = find_faces(img, face_model)
    #for face in faces:
    #found_face = True
    
    face = faces[0]
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
    

    cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
    cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    pitch = np.arctan2(Qx[2][1], Qx[2][2])
    roll = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
    yaw = np.arctan2(Qz[0][0], Qz[1][0])

    # tap.isPlaying:
    all_points_np = np.array([
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
        ])
    

    return img, image_points, all_points_np
    

    # print(dataset_np)


def get_face_dist(image_points):
    chin_y = image_points[1][1]
    right_mouth_corner_y = image_points[5][1]

    dist = abs(right_mouth_corner_y - chin_y)
    return dist

def handle_vlc(vlc_instance, stop_idle):

    if vlc_instance.is_playing() == 0:      
    
        if not stop_idle:
            media = vlc.Media("datensammeln/tap_loop_start0900-1050.mp4")
            vlc_instance.set_media(media)
            vlc_instance.play()
            print("start idle")
            print(stop_idle)

        else:
            if not stop_tap:
                print("**tapping sounds**")



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