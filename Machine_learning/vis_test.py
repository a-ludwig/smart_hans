from datenverarbeitung.dataloader import dataloader
from vis.visualizer import visualizer


dl = dataloader(scenario= 3, path="C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export", nr_taps=2, move_window_by=-10, feature_list= ["nosetip_y","nosetip_x","chin_x",
 "chin_y","left_eye_corner_x","left_eye_corner_y"])

# dl = dataloader(scenario= 3, path="C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export", nr_taps=1, move_window_by=-10, feature_list= ["nosetip_y","nosetip_x","chin_x",
# "chin_y","left_eye_corner_x","left_eye_corner_y","right_eye_corner_x","right_eye_corner_y","left_mouth_corner_x","left_mouth_corner_y","right_mouth_corner_x","right_mouth_corner_y","nose_end_point_x","nose_end_point_y"])
train, test = dl.get_train_test(frac = 0.8, seed = 0)

print(dl.df_labled)
vis = visualizer(dl = dl, path = 'vis/plots/', show = True)
vis.visualize(fname="scenario3/multi.png")