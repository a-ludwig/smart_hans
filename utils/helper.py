import numpy as np
import cv2
import time 
import os
from threading import Thread
import math

import socket

from Machine_learning.headpose_opencv.face_detector import get_face_detector, find_faces
from Machine_learning.headpose_opencv.face_landmarks import get_landmark_model, detect_marks

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

def init_hp_model():
    return get_face_detector(), get_landmark_model()

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


def get_camera_matrixes(img, rot_angle):
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


def connect_socket(ip, port):
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the Raspberry Pi
    s.connect((ip, port))
    return s 

def get_feedback(socket, number):
    # Send a number to the Raspberry Pi
    socket.sendall(str(number).encode())

    # Receive the response from the Raspberry Pi
    response = socket.recv(1024).decode()
    result = response

    # Close the socket
    socket.close()

    # Print the result
    print(f"The boolean value is {result}")
    return result