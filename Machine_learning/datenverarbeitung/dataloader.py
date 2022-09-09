import enum
import os
import numpy as np
import pandas as pd


class dataloader:
    def __init__(self, path, scenario, nr_taps = 1, move_window_by = 0, feature_list = []):
        self.path = path

        self.feature_list = feature_list

        self.scenario = scenario

        self.nr_taps = nr_taps
        self.move_window_by = move_window_by

        self.tap_size = 40

        self.window_size = self.nr_taps * self.tap_size
        self.df_len = 800

        #self.index_datapoint = index_datapoint
        self.univariate = False if len(feature_list) > 1 else True

        self.column_dict = {
            "nosetip_y":           0,
            "nosetip_x":           1 ,
            "chin_x":              2 ,
            "chin_y":              3 ,
            "left_eye_corner_x":   4 ,
            "left_eye_corner_y":   5 ,
            "right_eye_corner_x":  6 ,
            "right_eye_corner_y":  7 ,
            "left_mouth_corner_x": 8 ,
            "left_mouth_corner_y": 9 ,
            "right_mouth_corner_x":10,
            "right_mouth_corner_y":11,
            "nose_end_point_x":    12,
            "nose_end_point_y":    13,
            "head_pose1_x":        14,
            "head_pose1_y":        15,
            "head_pose2_x":        16,
            "head_pose2_y":        17,
            "jerk_expected":       18,
            "pitch":               19,
            "roll":                20,
            "yaw":                 21,
        }


        self.col_names = self.get_col_names(self.window_size)



        

    def get_train_test(self, frac, seed):
        
        self.tap_size = 40
        df_len = 800

        dataset_np = np.array([self.col_names])
        file_num = 1
        for file in os.listdir(self.path):

            print(file)
            start_annot = int(file.split("_")[5].split("-")[0]) 
            end_annot = int(file.split("_")[5].split("-")[1])

            target_tap_nr = int(start_annot/self.tap_size)

            file_arr = np.genfromtxt(self.path + '/' + file, skip_header=True, delimiter=',')
            #kürzen des arrays
            file_arr = file_arr[:df_len]
            feature_arr_list = []

            for elem in self.feature_list:
                index = self.column_dict[elem]

                feature_arr_list.append(file_arr[:,index])

            if self.scenario == 1 or self.scenario == 2:

                dataset_np = self.get_scenario_1_2(feature_arr_list, target_tap_nr, file, dataset_np, file_num)

            if self.scenario == 3:
                dataset_np = self.get_scenario_3(feature_arr_list, target_tap_nr, file, dataset_np)

            file_num = file_num + 1

        dataset_df  = pd.DataFrame(dataset_np[1:].tolist(), columns=self.col_names, dtype="float64")
        
        df_normalized = self.normalize_df(dataset_df)

        train, test = self.split_train_test(df = df_normalized.iloc[:, :-1], frac = frac, seed = seed)

        return train, test, df_normalized


    def get_scenario_1_2(self, feature_arr_list, target_tap_nr, file, dataset_np, file_num):
        """
        Scenario 1 limits the TS to a length of 800 frames. Each recording is split in 20 chunks of 40 Frames. 
        There are three classes: 0 = Before target, 1 = target, 2 = after target.
        The returned df has the shape (n, 41) where the first Column is the target class. 
        The data has been normalized.
            Paramters: 
                    path (str): location of CSV files
            Returns:
                    train (df), test (df)
        """

        """
        Scenario2 extends scenario1 with an extra class which is the the tap right after the target.
        This new class will replace class 2. Class 2 is now class 3.
            Paramters: 
                    path (str): location of CSV files
            Returns:
                    train (df), test (df)
        """
        for i in range(int(self.df_len/self.window_size)):

            #create array that only contains target value
            if self.scenario == 1:
                target = 0 if (i < target_tap_nr) else 1 if (i == target_tap_nr) else 2        

            if self.scenario == 2:
                target = 0 if (i < target_tap_nr) else 1 if (i == target_tap_nr) else 2 if (i == target_tap_nr + 1) else 3

            for i, elem in enumerate(feature_arr_list):

                window_arr = elem[i*self.window_size:(i+1)*self.window_size]

                labeled_window = self.get_labeled_window(target, file_num, i, window_arr, file)

                dataset_np = self.stack_dataset(dataset_np, labeled_window)

        return dataset_np

    def get_scenario_3(self, arr, target_tap_nr, file, dataset_np ):
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

            for i in range(self.nr_taps):
                #change itterator depending on class
                if k == 0:
                    #reverse itterator
                    j = -(self.nr_taps - i)
                if k == 1:
                    j = i

                #define delimeter 
                start_del = (target_tap_nr + j + 1) * self.tap_size + self.move_window_by
                end_del = (target_tap_nr + j + 2) * self.tap_size + self.move_window_by
                #fill temp arr and append to target_class_array
                temp_arr = arr[start_del : end_del]
                target_class_arr = np.append(target_class_arr, temp_arr)
            target_class_arr = np.append(target_class_arr, [file[:-4]])

            #check length of arr to make sure that files that are too short still can be used
            if len(target_class_arr) == len(self.col_names):
                dataset_np = np.vstack([dataset_np, target_class_arr])
        return dataset_np

    def get_col_names(self, window_size):
        if self.univariate == True:
            col_names =  ['target']
        else :
            col_names = ['index','feature']

        for i in range(window_size):
            col_names.append(i)
        
        if self.univariate == False:
            col_names.append('target')
        col_names.append('file_name')
        return col_names

    def normalize_df(self, df):
        if self.univariate == True:
            start_del = 1
            end_del = -1
        else:
            start_del = 2
            end_del = -2
        df_max_scaled = df.copy()
        #get max value of all columns
        max = df_max_scaled.iloc[:, start_del:end_del].abs().max().max()
        # apply normalization techniques
        for column in df_max_scaled.iloc[:, start_del:end_del].columns:
            df_max_scaled[column] = df_max_scaled[column].abs()  / max

        return df_max_scaled

    def split_train_test(self, df, frac = 0.8, seed = 0):
        train = df.sample(frac=frac,random_state=seed)
        test = df.drop(train.index)
        return train, test

    def get_labeled_window(self, target, file_num, feature, window_arr, file):
        t_arr = np.array([target])
        i_arr = np.array([file_num])
        i_f_arr = np.append(i_arr, feature+1)#index then feature(starting with 1)

        if self.univariate:
            label_arr = t_arr
        else:
            label_arr = i_f_arr
        labeled_arr = np.append(label_arr, window_arr)
        if not self.univariate:
            labeled_arr = np.append(labeled_arr, t_arr)

        labeled_arr = np.append(labeled_arr, [file[:-4]])#filename without csv

        return labeled_arr

    def stack_dataset(self, dataset_np, labeled_window):
        #check length of arr to make sure that files that are too short still can be used
        if len(labeled_window) == len(self.col_names):
            dataset_np = np.vstack([dataset_np, labeled_window])
        return dataset_np