from datenverarbeitung.scenario3 import dataloader
from vis import vis_data


dl = dataloader(scenario= 2, path="C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/gesammelt", nr_taps=1, move_window_by=-10)
train, test, df_labled = dl.get_train_test(frac = 0.8, seed = 0)

print(df_labled)
vis_data.visualize(df = df_labled, fname = 'vis/plots/scenario3.png', show = True)
