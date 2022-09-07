from datenverarbeitung.dataloader import dataloader
from vis import vis_data


dl = dataloader(scenario= 1, path="/home/adi/cloudy_adlu/smart_hans/AP2/Daten/headpose_opencv_pitch_roll_yaw_20220904", nr_taps=1, move_window_by=-10, index_datapoint = 2, univariate=True)
train, test, df_labled = dl.get_train_test(frac = 0.8, seed = 0)

print(df_labled)
vis_data.visualize(df = df_labled, fname = 'vis/plots/scenario3.png', show = True)
