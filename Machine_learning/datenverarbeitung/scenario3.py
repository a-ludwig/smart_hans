import os
import numpy as np
import pandas as pd



def get_train_test(path, scenario, nr_taps, move_window_by = 0):
    
    tap_size = 40
    window_size = nr_taps * tap_size
    df_len = 800
    
    col_names = get_col_names(window_size)

    dataset_np = np.array([col_names])
    file_num = 1
    for file in os.listdir(path):

        print(file)
        start_annot = int(file.split("_")[5].split("-")[0]) 
        end_annot = int(file.split("_")[5].split("-")[1])

        target_tap_nr = int(start_annot/tap_size)

        file_np = np.genfromtxt(path + '/' + file, skip_header=True, delimiter=',')
        nosetip_np = file_np[:df_len,2]

        #########
        #Class 0#
        #########
        if scenario == 3:
            dataset_np = get_scenario_3(nosetip_np, nr_taps, target_tap_nr, tap_size, move_window_by, file, col_names, dataset_np )
    

    dataset_df  = pd.DataFrame(dataset_np[1:].tolist(), columns=col_names, dtype="float64")
    
    df_normalized = normalize_df(dataset_df)

    train, test = split_train_test(df = df_normalized.iloc[:, :-1], frac = 0.8, seed = 0)

    return train, test, df_normalized

def get_scenario_3(arr, nr_taps, target_tap_nr, tap_size, move_window_by, file, col_names, dataset_np ):
    """
    Scenario3 splits one recording in two classes and only two TS.
    Class 0: nr_taps-1 taps before and the target
    Class 1: the nr_taps taps after the target
        Paramters: 
                path (str): location of CSV files
                nr_taps (int): nr of relevant taps (only scenario 3)
                move_window_by (int): moves the tap window by given amount
        Returns:
                train (df), test (df), full_labled(df)
    """
    for k in range(2):
        #create array that only contains target value
        target_class_arr = np.array([k])

        for i in range(nr_taps):
            #change itterator depending on class
            if k == 0:
                #reverse itterator
                j = -(nr_taps - i)
            if k == 1:
                j = i

            #define delimeter 
            start_del = (target_tap_nr + j + 1) * tap_size + move_window_by
            end_del = (target_tap_nr + j + 2) * tap_size + move_window_by
            #fill temp arr and append to target_class_array
            temp_arr = arr[start_del : end_del]
            target_class_arr = np.append(target_class_arr, temp_arr)
        target_class_arr = np.append(target_class_arr, [file[:-4]])

        #check length of arr to make sure that files that are too short still can be used
        if len(target_class_arr) == len(col_names):
            dataset_np = np.vstack([dataset_np, target_class_arr])
    return dataset_np

def get_col_names(window_size):
    col_names =  ['target']

    for i in range(window_size):
        col_names.append(i)
    col_names.append('file_name')
    return col_names

def normalize_df(df):
    df_max_scaled = df.copy()
    #get max value of all columns
    max = df_max_scaled.iloc[:, 1:-1].abs().max().max()
    # apply normalization techniques
    for column in df_max_scaled.iloc[:, 1:-1].columns:
        df_max_scaled[column] = df_max_scaled[column]  / max
    return df_max_scaled

def split_train_test(df, frac = 0.8, seed = 0):
    train = df.sample(frac=frac,random_state=seed)
    test = df.drop(train.index)
    return train, test