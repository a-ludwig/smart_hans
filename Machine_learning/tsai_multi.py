from turtle import shape
from tsai.all import *
import pandas as pd
from datenverarbeitung.dataloader import dataloader

num_scenario = 3
dl = dataloader(scenario= num_scenario, path="C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/single_file_for_testing", nr_taps=1, move_window_by=-10, feature_list = ["nosetip_y", "pitch", "nosetip_x"])
df_n = dl.get_train_test(frac = 0.8, seed = 0)



cols = list(df_n.columns)
a, b = cols.index('index'), cols.index('feature')
cols[b], cols[a] = cols[a], cols[b]
df_n = df_n[cols]
print(df_n)
df_n = df_n.sort_values(['target','index'])

### Debugging Data for Comparison
ds_name = 'OliveOil'
X_UCR, y_UCR, _ = get_UCR_data(ds_name, return_split=False)
X_UCR = X_UCR[:, 0]
y_UCR = y_UCR.reshape(-1, 1)
data = np.concatenate((X_UCR, y_UCR), axis=-1)
df = pd.DataFrame(data).astype(float)
df = df.rename(columns={570: 'target'})
df1 = pd.concat([df, df + 10, df + 100], axis=0).reset_index(drop=False)
df2 = pd.DataFrame(np.array([1] * 60 + [2] * 60 + [3] * 60), columns=['feature'])
df_UCR = pd.merge(df2, df1, left_index=True, right_index=True)

#df = pd.concat([train, test])
print(df_n.head())
print(df_UCR.head())

X_UCR, y_UCR = df2xy(df_UCR, sample_col='index', feat_col='feature', target_col='target', data_cols=None)

data_cols = dl.col_names[2:-2]
X, y = df2xy (df_n, sample_col='index', feat_col='feature', target_col='target', data_cols=data_cols)
#print("y")
#print(y_UCR)

#X, y, _ = get_UCR_data("/home/adi/cloudy_adlu/smart_hans/AP2/Daten/single_file_for_testing/", return_split=False)
print(f"shape ours {df_n.shape}")
print(f"shape UCR {df_UCR.shape}")

print(X_UCR.shape)
print(y_UCR)
# X = np.swapaxes(X, 0,1)
# y = np.swapaxes(y, 0,1)
print(X.shape)
print(y)

splits = get_splits(y, valid_size=.2)


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X,y, tfms=[None, TSClassification()], inplace=True, splits=splits)

dsets

#dls = TSDataLoaders.from_numpy(X, y)

dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

dls.show_batch(sharey=True)