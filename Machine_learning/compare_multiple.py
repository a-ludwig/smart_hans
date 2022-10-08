from tsai.all import *
import pandas as pd
from datenverarbeitung.dataloader import dataloader
from IPython.display import clear_output
import datetime


num_scenario = 3
nr_taps = 2
model_to_use = "InceptionTimePlus"
learning_cycles = 1
features_to_learn_with = ["nosetip_y"]
feature_list_string = '_'.join(features_to_learn_with)
models_folder = "models"
plots_folder = "vis/plots"
save_name = "scenario_{}_{}".format(num_scenario, model_to_use)


dl = dataloader(scenario= num_scenario, path="/home/adi/cloudy_adlu/smart_hans/AP2/Daten/headpose_opencv_pitch_roll_yaw_20220904", nr_taps=nr_taps, move_window_by=-10, feature_list=features_to_learn_with)
train, test= dl.get_train_test(frac = 0.8, seed = 0)

X_test, y_test = df2xy(test, target_col='target')
X_train, y_train = df2xy(train, target_col='target')

X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets

dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

archs = [(FCN, {}), (ResNet, {}), (xresnet1d34, {}), (ResCNN, {}), 
        (LSTM, {'n_layers':1, 'bidirectional': False}), (LSTM, {'n_layers':2, 'bidirectional': False}), (LSTM, {'n_layers':3, 'bidirectional': False}), 
        (LSTM, {'n_layers':1, 'bidirectional': True}), (LSTM, {'n_layers':2, 'bidirectional': True}), (LSTM, {'n_layers':3, 'bidirectional': True}),
        (LSTM_FCN, {}), (LSTM_FCN, {'shuffle': False}), (InceptionTime, {}), (XceptionTime, {}), (OmniScaleCNN, {}), (mWDN, {'levels': 4})]

results = pd.DataFrame(columns=['arch', 'hyperparams',  'train loss', 'valid loss', 'accuracy', 'time'])
for i, (arch, k) in enumerate(archs):

    save_name = "scenario_{}_{}_features_{}".format(num_scenario, arch.__name__,feature_list_string)
    ## set parameters for modelsaves
    scenario_name_stage0 = save_name+"_nr_taps_{}".format(str(nr_taps))+"_stage0"

    model = create_model(arch, dls=dls, **k)
    print(model.__class__.__name__)
    learn = Learner(dls, model,  metrics=accuracy)

    learn.save(scenario_name_stage0)
    learn.load(scenario_name_stage0)
    learn.lr_find()
    scenario_name_stage1 = scenario_name_stage0.replace("0","1")

    start = time.time()
    learn.fit_one_cycle(learning_cycles, 1e-3)
    elapsed = time.time() - start
    vals = learn.recorder.values[-1]
    results.loc[i] = [arch.__name__, k, vals[0], vals[1], vals[2], int(elapsed)]
    results.sort_values(by='accuracy', ascending=False, ignore_index=True, inplace=True)

    learn.plot_confusion_matrix()
    current_time= datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
    plot_name = plots_folder+"/confusion_matrix_"+save_name+"_nrtaps_{}_features_{}_learning_cycles_{}_{}.png".format(str(nr_taps),feature_list_string,learning_cycles, current_time)
    plt.savefig(plot_name, ext='png', bbox_inches="tight")
    clear_output(wait=True)
    display(results)
