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

from Machine_learning.headpose_opencv.face_detector import get_face_detector, find_faces
from Machine_learning.headpose_opencv.face_landmarks import get_landmark_model, detect_marks
from Machine_learning.datenverarbeitung.dataloader import dataloader
from utils.horse import horse
import pandas as pd
import numpy as np


from tsai.all import *
from threading import Thread

rot_angle = 0 #270
debug = True
found_face = False
stop_idle = False
stop_tap = False
curr_num = 1
df = pd.DataFrame()

##threshold for distance detection
thresh = 35
##waiting time for Hansi to Start in Secs
waiting_time = 5


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
class VideoStreamWidget(object): ###https://stackoverflow.com/questions/54933801/how-to-increase-performance-of-opencv-cv2-videocapture0-read#55131847
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

def main():
    
    nr_taps = 1
    tap_size = 30
    window_size = tap_size * nr_taps
    move_by = -12
    prediction_threshold = 0.571

    cycle_size = 23
    n = 2 ### cycle/n for modulo

    min = 0
    max = 0
    #load tsai model
    predictor = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    #load hp model
    face_model = get_face_detector()
    landmark_model = get_landmark_model()

    hansi = horse() ##hansi is our magnificent horse 

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
    while True:
        try:
            #video_stream_widget.show_frame()
            img = video_stream_widget.frame
            fps = FPS().start()

            img = cv2.warpAffine(img, rot_M, (img.shape[1], img.shape[0]))## rot image
            hansi.queue()## keep hansi alive

            faces, face_found = find_face(img, face_model)
            
            if face_found:
                img, image_points, all_points = estimate_head_pose(img, model_points, cam_M, face_model, landmark_model, faces)

                dist = get_face_dist(image_points)
                #print(f"hansi curr tap: {hansi.curr_tap}")
                if hansi.switch == "tapping" and hansi.curr_tap >= 2:
                    data.append(all_points)
                    hansi.curr_win_size += 1
                    
                    delim = hansi.curr_win_size ##+move_by ###movy_by only negative, otherwise: (len(data) + dl.move_window_by) if dl.move_window_by >=0 else len(data) 
                    #print(delim)
                    #######
                    #Make prediction  n times per cycle(tap)
                    #######
                    #print(hansi.curr_tap)
                    if (delim % (cycle_size+move_by) == 0)and hansi.curr_tap >= 3: #delim % (cycle_size/n) == 0 and hansi.curr_tap >= 3: 

                        print(f"im predicting at calc_tap:{hansi.curr_tap}")
                        #print(delim)
                        #print("***predicting***")
                        window_scaled, min, max = list_to_norm_win(dl, data, min, max)
                        predicted_class = make_pred(window_scaled, predictor, threshold=prediction_threshold, class_to_look_at=1)
                        #print(f"predicted class: {predicted_class}")
                        if predicted_class == 1:
                            hansi.switch = "end_tap"
                            hansi.save = True
                            

                            #### reset for new participant

            else:
                hansi.switch = "idle"
                dist = 0
                curr_win_size = 0
            

            timer_in_sec, last_t = wait_for_face(timer_in_sec, last_t, dist, thresh)

            if int(timer_in_sec) == waiting_time and hansi.switch == "idle":
                hansi.switch = "start_tap"
            elif timer_in_sec < waiting_time and hansi.switch == "tapping": 
                hansi.switch = "end_tap"

            if hansi.switch == "reset_idle":
                if hansi.save == True:
                    df2 = pd.DataFrame(data)
                    now = archi.now()
                    date_time = now.strftime("%m%d%Y_%H%M%S")

                    df2.to_csv(f"installation_export/inst_exp_{date_time}.csv")
                    hansi.save = False
                timer_in_sec, last_t, data = init_params(num_params)

            
            # update the FPS counter
            fps.update()
            fps.stop()


            color = (0,255,0) if stop_idle else (255,0,0)

            cv2.putText(img, str(int(timer_in_sec)), [100,100], font, 2, color, 3)
            cv2.putText(img, str(dist), [180,100], font, 2, color, 3)
            cv2.putText(img, str(int(fps.fps())), [100,200], font, 2, (0,0,255), 3)


            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except AttributeError:
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
    

def get_face_dist(image_points):
    chin_y = image_points[1][1]
    right_mouth_corner_y = image_points[5][1]

    dist = abs(right_mouth_corner_y - chin_y)
    return dist

if __name__ == "__main__":
    main()