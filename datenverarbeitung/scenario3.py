import os
import numpy as np
import pandas as pd

def get_scenario_3(path):
    """
    Scenario3 splits one recording in two classes and only two TS.
    Class 0: the tap before and the target
    Class 1: the two taps after the target
        Paramters: 
                path (str): location of CSV files
        Returns:
                train (df), test (df)
    """
    tap_size = 40
    window_size = 80
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

        anno_df_num = int(start_annot/tap_size)

        file_np = np.genfromtxt(path + '/' + file, skip_header=True, delimiter=',')
        nosetip_np = file_np[:df_len,2]

        #########
        #Class 0#
        #########
        #create array that only contains target value
        target_arr = np.array([0])
        temp_np = nosetip_np[(anno_df_num-1) * tap_size : (anno_df_num+1) * tap_size ]
        temp_np = np.append(target_arr, temp_np)
            
        #check length of arr to make sure that files that are too short still can be used
        if len(temp_np) == len(col_names):
            dataset_np = np.vstack([dataset_np, temp_np])

        #########
        #Class 1#
        #########
        #create array that only contains target value
        target_arr = np.array([1])
        temp_np = nosetip_np[(anno_df_num+1) * tap_size : (anno_df_num+3) * tap_size ]
        temp_np = np.append(target_arr, temp_np)

        if len(temp_np) == len(col_names):
            dataset_np = np.vstack([dataset_np, temp_np])


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
