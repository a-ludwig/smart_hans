import os
from turtle import shape
import numpy as np
import pandas as pd
import csv


def get_scenario_1(path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export"):
    """
    Scenarion1 limits the TS to a length of 800 frames. Each recording is split in 20 chunks of 40 Frames. 
    There are three classes: 0 = Before target, 1 = target, 2 = after target.
    The returned df has the shape (n, 41) where the first Column is the target class. 
    The data has been normalized.
        Paramters: 
                path (str): location of CSV files
        Returns:
                train (df), test (df)
    """

    window_size = 40
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

        anno_df_num = int(start_annot/window_size)
        
        file_np = np.genfromtxt(path + '/' + file, skip_header=True, delimiter=',')
        nosetip_np = file_np[:df_len,2]
        
        for i in range(int(df_len/window_size)):

            index = int(file_num * i)
            #create array that only contains target value
            target = 0 if (i < anno_df_num) else 1 if (i == anno_df_num) else 2
            arr = np.array([target])

            temp_np = nosetip_np[i*window_size:(i+1)*window_size]
            temp_np = np.append(arr, temp_np)
            
            #check length of arr to make sure that files that are too short still can be used
            if len(temp_np) == 41:
                dataset_np = np.vstack([dataset_np, temp_np])

        file_num = file_num + 1

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