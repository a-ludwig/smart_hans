import os
from turtle import shape
import numpy as np
import pandas as pd
import csv


def get_szenario_1(path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export"):
    """
    Szenarion1
    Diese Funktion gibt ein training und test Datensatz zurück.
    Dafür wird die TS auf 800 Frames beschränkt und eine Windowsize von 40 gewählt
        Paramters: 
                path (str): Ort der CSV-Dateien
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
            target = 0 if (i < anno_df_num) else 1 if (i == anno_df_num) else 2
            arr = np.array([target])

            temp_np = nosetip_np[i*window_size:(i+1)*window_size]
            temp_np = np.append(arr, temp_np)

            dataset_np = np.vstack([dataset_np, temp_np])

        file_num = file_num + 1

    dataset_df  = pd.DataFrame(dataset_np[1:].tolist(), columns=col_names)

    train = dataset_df.sample(frac=0.8,random_state=0)
    test = dataset_df.drop(train.index)
    return train, test