#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import pandas as pd
import argparse
import numpy as np

#import Face Recognition libraries
import mediapipe as mp
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks


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

def process_single_video(path, file, rot_angle, predictor, face_detection, mp_drawing, world, face3Dmodel, out) :
    data = []
    print(file)
    start_annot = int(file.split("_")[5].split("-")[0])
    end_annot = int(file.split("_")[5].split("-")[1])
    cap = cv2.VideoCapture(path+"/"+file)
    ret, img = cap.read()

    #rotate image
    size = img.shape
    center = (size[1]/2, size[0]/2)
    M = cv2.getRotationMatrix2D(center, rot_angle, scale = 1)
    img = cv2.warpAffine(img, M, (size[1], size[0]))

    size = img.shape
    # Camera internals
    center = (size[1]/2, size[0]/2)
    focal_length = size[1]
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    refImgPts = None
    while True:
        ret, img = cap.read()
        if ret == True: 
            size = img.shape
            img = cv2.warpAffine(img, M, (size[1], size[0]))
            fps = cap.get(cv2.CAP_PROP_FPS)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            h, w, c = image.shape
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not ret:
                print(f'[ERROR - System]Cannot read from source: {args["camsource"]}')
                break
            if results.detections: 
                for detection in results.detections:
                        print(results.detections)
                        print(detection)
                        location = detection.location_data
                        relative_bounding_box = location.relative_bounding_box
                        x_min = relative_bounding_box.xmin
                        y_min = relative_bounding_box.ymin
                        widthh = relative_bounding_box.width
                        heightt = relative_bounding_box.height
                        if mp_drawing._normalized_to_pixel_coordinates(x_min,y_min,w,h) :
                            absx, absy = mp_drawing._normalized_to_pixel_coordinates(x_min,y_min,w,h)
                        if mp_drawing._normalized_to_pixel_coordinates(x_min+widthh,y_min+heightt,w,h) :
                            abswidth,absheight = mp_drawing._normalized_to_pixel_coordinates(x_min+widthh,y_min+heightt,w,h)
                        else :
                            absx = absx
                            absy = absy
                            abswidth = abswidth
                            absheight = absheight
                        
                        newrect = dlib.rectangle(absx,absy,abswidth,absheight)
                        cv2.rectangle(image, (absx, absy), (abswidth, absheight),
                        (0, 255, 0), 2)
                        shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)


                # draw(image, shape)

                        refImgPts = world.ref2dImagePoints(shape)

                        height, width, channels = img.shape
                        focalLength = 1.0 * width
                        #focalLength = args["focal"] * width
                        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

                        mdists = np.zeros((4, 1), dtype=np.float64)

                        # calculate rotation and translation vector using solvePnP
                        success, rotationVector, translationVector = cv2.solvePnP(
                            face3Dmodel, refImgPts, cameraMatrix, mdists)

                        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
                        noseEndPoint2D, jacobian = cv2.projectPoints(
                            noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

                        #  draw nose line
                        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
                        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
                        cv2.line(image, p1, p2, (110, 220, 0),
                                thickness=2, lineType=cv2.LINE_AA)
                        

                        # calculating euler angles
                        rmat, jac = cv2.Rodrigues(rotationVector)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                        print('*' * 80)
                        # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
                        pitch = np.arctan2(Qx[2][1], Qx[2][2])
                        roll = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
                        yaw = np.arctan2(Qz[0][0], Qz[1][0])
                        

                        p1 = ( int(refImgPts[0][0]), int(refImgPts[0][1]))
                        p2 = ( int(noseEndPoint2D[0][0][0]), int(noseEndPoint2D[0][0][1]))
                        x1, x2 = head_pose_points(img, rotationVector, translationVector, cameraMatrix)
                    #cv2.putText(image, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv2.imshow("Head Pose", image)

        #data.append((x,y,z))

            annotate = 0 
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame >= start_annot and current_frame <= end_annot :
                #print(" i will annotate at frame "+ str(current_frame))
                annotate = 1
            annotate=1
            if results.detections :
                if refImgPts.any() :
                    data.append( {
                                "nosetip_x" : refImgPts[0][0],
                                "nosetip_y" : refImgPts[0][1],
                                "chin_x" : refImgPts[1][0],
                                "chin_y" : refImgPts[1][1],
                                "left_eye_corner_x" : refImgPts[2][0],
                                "left_eye_corner_y" : refImgPts[2][1],
                                "right_eye_corner_x" : refImgPts[3][0],
                                "right_eye_corner_y" : refImgPts[3][1],
                                "left_mouth_corner_x" : refImgPts[4][0],
                                "left_mouth_corner_y" : refImgPts[4][1],
                                "right_mouth_corner_x" : refImgPts[5][0],
                                "right_mouth_corner_y" : refImgPts[5][1],
                                "nose_end_point_x" : p2[0],
                                "nose_end_point_y" : p2[1],
                                "head_pose1_x": x1[0],
                                "head_pose1_y": x1[1],
                                "head_pose2_x": x2[0], 
                                "head_pose2_y": x2[1],
                                "jerk_expected" : annotate,
                                "pitch": pitch,
                                "roll": roll,
                                "yaw": yaw,
                            } )

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            #
        else:
                #print(data)
                print(file + " processed")
                df2 = pd.DataFrame(data)
                df2.to_csv(out+"/"+file.split('.')[0]+file.split('.')[1]+".csv")           
                break
    cv2.destroyAllWindows()
    cap.release()

    return

def main(args):

    # helper modules
    from drawFace import draw
    import reference_world as world

    #Settingup MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        min_detection_confidence=0.5)

    PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

    if not os.path.isfile(PREDICTOR_PATH):
        print("[ERROR] USE models/downloader.sh to download the predictor")
        sys.exit()


    face3Dmodel = world.ref3DModel()
 #####
    rot_angle = 270
#####

    print(args)
    if len(args) >=3 :   
        path =args[1]
        out = args[2]
    else : 
        path = "/home/adi/Videos/Hansi_daten/smart_hans_rot_angle_270"
        out = "preprocessed_debug"
        print("please add a path for input and output to running this script")
        #exit()

    if not os.path.exists(out):
       os.makedirs(out)

    predictor = dlib.shape_predictor(PREDICTOR_PATH)

   # cap = cv2.VideoCapture(args["camsource"])
    #cap = cv2.VideoCapture(0)
    df = pd.DataFrame()
    data = [] 
    counter = 0
    for file in os.listdir(path) :
        out_file = file.replace(".", "")
        out_file = out + "/" + out_file.replace("avi", ".csv")
        print(out_file)
        if os.path.isfile(out_file):
            counter = counter + 1
            print(out_file + " already exists")
        else:
            process_single_video(path, file, rot_angle, predictor, face_detection, mp_drawing, world, face3Dmodel, out)
    print("done")
    print(len(os.listdir(out)))
    print ("files already processed: " + str(counter))
    print ("total files: " + str(len(os.listdir(path))))


if __name__ == "__main__":
    # path to your video file or camera serial
    main(sys.argv)
