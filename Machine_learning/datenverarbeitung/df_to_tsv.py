
from dataloader import dataloader


dl = dataloader(scenario= 3, path="C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/headpose_opencv_pitch_roll_yaw_20220904", nr_taps=3, move_window_by=0, feature_list= ["nosetip_y"])
train, test = dl.get_train_test(frac = 0.8, seed = 0)

print(test)
print(train)



test.to_csv('C:/Users/peter/Documents/Projekte/Hans/transfer_learning/archives/TSC/example/example_test.tsv', sep="\t", index= False, header= False)
train.to_csv('C:/Users/peter/Documents/Projekte/Hans/transfer_learning/archives/TSC/example/example_train.tsv', sep="\t", index= False, header= False)