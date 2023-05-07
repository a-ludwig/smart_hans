from datenverarbeitung.dataloader import dataloader
from vis.visualizer import visualizer
import os

dl = dataloader(scenario= 3, path="C:/Users/adi/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export", nr_taps=1, move_window_by=-10,tap_size=40, feature_list= ["nosetip_y","nosetip_x"] )

# dl = dataloader(scenario= 3, path="C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export", nr_taps=1, move_window_by=-10, feature_list= ["nosetip_y","nosetip_x","chin_x",
# "chin_y","left_eye_corner_x","left_eye_corner_y","right_eye_corner_x","right_eye_corner_y","left_mouth_corner_x","left_mouth_corner_y","right_mouth_corner_x","right_mouth_corner_y","nose_end_point_x","nose_end_point_y"])
train, test = dl.train, dl.test
print(dl.df)

#print(dl.df_labled)
vis = visualizer(dl = dl, path = 'C:/Users/adi/Documents/code/smart_hans/Machine_learning/vis/plots/doku', show=False)
#for file in os.listdir("C:/Users/adi/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export"):
#   vis.visualize(file)
vis.visualize_raw(path='C:/Users/adi/Nextcloud/smart_hans/AP2/Daten/headpose_opencv_pitch_roll_yaw_20220904')