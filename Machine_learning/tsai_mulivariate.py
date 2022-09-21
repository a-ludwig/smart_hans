from turtle import shape
from tsai.all import *
import pandas as pd
from datenverarbeitung.dataloader import dataloader

num_scenario = 1
dl = dataloader(scenario= num_scenario, path="/home/adi/cloudy_adlu/smart_hans/AP2/Daten/single_file_for_testing", nr_taps=1, move_window_by=-10, feature_list = ["nosetip_y", "pitch"])
train, test = dl.get_train_test(frac = 0.8, seed = 0)

X_train, y_train = df2xy (train, sample_col='index', feat_col='feature', target_col='target', data_cols=None)
X_test, y_test = df2xy (test, sample_col='index', feat_col='feature', target_col='target', data_cols=None)


ds_name = 'OliveOil'
X, y, _ = get_UCR_data(ds_name, return_split=False)

print(X.shape)
print(y.shape)
#print(splits)

# X_train, y_train = df2xy(train_df, target_col='target')
# np.shape(X)
# test_eq(X_test.shape, (60, 1, 40))
# test_eq(y_test.shape, (60, ))

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets

dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

dls.show_batch(sharey=True)