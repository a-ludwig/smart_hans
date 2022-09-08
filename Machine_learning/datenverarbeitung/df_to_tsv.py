
from dataloader import da


dl = dataloader(scenario= 1, path="C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/headpose_opencv_pitch_roll_yaw_20220904", nr_taps=1, move_window_by=-10, feature_list= ["nosetip_x", "pitch"], univariate=True)
train, test, df_labled = dl.get_train_test(frac = 0.8, seed = 0)

print(test)
print(train)



test.to_csv('C:/Users/peter/Documents/Projekte/HANS/matlab/example_test.tsv', sep="\t", index= False, header= False)
train.to_csv('C:/Users/peter/Documents/Projekte/HANS/matlab/example_train.tsv', sep="\t", index= False, header= False)