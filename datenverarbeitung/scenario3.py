import os
import numpy as np
import pandas as pd

def get_scenario_3(path, nr_taps, move_window_by = 0):
    """
    Scenario3 splits one recording in two classes and only two TS.
    Class 0: nr_taps-1 taps before and the target
    Class 1: the nr_taps taps after the target
        Paramters: 
                path (str): location of CSV files
                nr_taps (int): nr of relevant taps
                move_window_by (int): moves the tap window by given amount
        Returns:
                train (df), test (df)
    """
    tap_size = 40
    window_size = nr_taps * tap_size
    df_len = 800
    col_names =  ['target']

    for i in range(window_size):
        col_names.append(i)

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
        #create array that only contains target value
        target_class_arr = np.array([0])

        for i in range(nr_taps):
            #reverse itterator
            j = nr_taps - i - 1
            #define delimeter for class 0
            start_del = (target_tap_nr - j) * tap_size + move_window_by
            end_del = (target_tap_nr - j + 1) * tap_size + move_window_by
            #fill temp arr and append to target_class_array
            temp_arr = nosetip_np[start_del : end_del]
            target_class_arr = np.append(target_class_arr, temp_arr)
            
        #check length of arr to make sure that files that are too short still can be used
        if len(target_class_arr) == len(col_names):
            dataset_np = np.vstack([dataset_np, target_class_arr])

        #########
        #Class 0#
        #########
        #create array that only contains target value
        target_class_arr = np.array([1])

        for i in range(nr_taps):
            #define delimeter for class 0
            start_del = (target_tap_nr + i + 1) * tap_size + move_window_by
            end_del = (target_tap_nr + i + 2) * tap_size  + move_window_by
            #fill temp arr and append to target_class_array
            temp_arr = nosetip_np[start_del : end_del]
            target_class_arr = np.append(target_class_arr, temp_arr)
            
        #check length of arr to make sure that files that are too short still can be used
        if len(target_class_arr) == len(col_names):
            dataset_np = np.vstack([dataset_np, target_class_arr])


    dataset_df  = pd.DataFrame(dataset_np[1:].tolist(), columns=col_names, dtype="float64")

    
    df_max_scaled = dataset_df.copy()
    #get max value of all columns
    max = df_max_scaled.iloc[:, 1:].abs().max().max()
    # apply normalization techniques
    for column in df_max_scaled.iloc[:, 1:].columns:
        df_max_scaled[column] = df_max_scaled[column]  / max
        

    train = df_max_scaled.sample(frac=0.8,random_state=0)
    test = df_max_scaled.drop(train.index)
    return train, test
