from cmath import nan
import enum
import os
import numpy as np
import pandas as pd


class dataloader:
    def __init__(self, path="leer", scenario=3, nr_taps = 1, move_window_by = 0, feature_list = [], tap_size = 40, frac = 0.7):
        self.path = path

        self.feature_list = feature_list

        self.scenario = scenario

        self.nr_taps = nr_taps
        self.move_window_by = move_window_by

        self.tap_size = tap_size

        self.file_num = 0

        self.frac = frac

        self.window_size = nr_taps * self.tap_size
        self.df_len = 800

        self.df_labled = 1

        self.df = None
        self.train = None
        self.test = None
        #self.index_datapoint = index_datapoint
        self.univariate = False if len(feature_list) > 1 else True

        self.column_dict = {
            "nosetip_y":           1 ,
            "nosetip_x":           2 ,
            "chin_x":              3 ,
            "chin_y":              4 ,
            "left_eye_corner_x":   5 ,
            "left_eye_corner_y":   6 ,
            "right_eye_corner_x":  7 ,
            "right_eye_corner_y":  8 ,
            "left_mouth_corner_x": 9 ,
            "left_mouth_corner_y": 10,
            "right_mouth_corner_x":11,
            "right_mouth_corner_y":12,
            "nose_end_point_x":    13,
            "nose_end_point_y":    14,
            "head_pose1_x":        15,
            "head_pose1_y":        16,
            "head_pose2_x":        17,
            "head_pose2_y":        18,
            "jerk_expected":       19,
            "pitch":               20,
            "roll":                21,
            "yaw":                 22
        }

        self.col_names = self.get_col_names(self.window_size)

        self.get_train_test(self.frac, seed = 420)


    def get_train_test(self, frac, seed):
        
        #self.tap_size = 40
        df_len = 800

        dataset_np = np.array([self.col_names])
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

                dataset_np = self.get_scenario_1_2(feature_arr_list, target_tap_nr, file, dataset_np)

            if self.scenario == 3:
                dataset_np = self.get_scenario_3(feature_arr_list, target_tap_nr, file, dataset_np)

#            self.file_num = self.file_num + 1

        dataset_df  = pd.DataFrame(dataset_np[1:].tolist(), columns=self.col_names, dtype="float64")
        
        #df_normalized = self.normalize_df_by_feature(dataset_df)
        df_normalized = self.normalize_df_by_window(dataset_df)

        self.df_labled = df_normalized

        self.train, self.test = self.split_train_test(df = df_normalized.iloc[:, :-1], frac = frac, seed = seed)
        self.df = df_normalized.iloc[:, :-1]

        return 


    def get_scenario_1_2(self, feature_arr_list, target_tap_nr, file, dataset_np):
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

            for j, elem in enumerate(feature_arr_list):

                window_arr = elem[i*self.window_size:(i+1)*self.window_size]

                labeled_window = self.get_labeled_window(target, self.file_num, j, [window_arr], file)

                dataset_np = self.stack_dataset(dataset_np, labeled_window)
            self.file_num = self.file_num +1

        return dataset_np

    def get_scenario_3(self, feature_arr_list, target_tap_nr, file, dataset_np ):
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
        for target in range(2):
            for k, elem in enumerate(feature_arr_list):
                temp_arr = np.array([])
                window_list = []
                for i in range(self.nr_taps):
                    #change itterator depending on class
                    if target == 0:
                        #reverse itterator
                        j = -(self.nr_taps - i)
                    if target == 1:
                        j = i
                    # if k == 0:
                    #     new_target = 0 + target + 1 
                    # else:
                    #     new_target = pow(10, k) + target+1
                    new_target = target
                    #define delimeter 
                    start_del = (target_tap_nr + j + 1) * self.tap_size + self.move_window_by
                    end_del = (target_tap_nr + j + 2) * self.tap_size + self.move_window_by

                    #fill temp arr and append to target_class_array
                    window_arr = elem[start_del : end_del]
                    window_list.append(window_arr)

                    
                labeled_window = self.get_labeled_window(new_target, self.file_num, k , window_list, file)
                dataset_np = self.stack_dataset(dataset_np, labeled_window)
            self.file_num = self.file_num +1
        return dataset_np

    def get_col_names(self, window_size):
        if self.univariate == True:
            col_names =  ['target']
        else :
            col_names = ['sample_index','feature']

        for i in range(window_size):
            col_names.append(i)
        
        if self.univariate == False:
            col_names.append('target')
        col_names.append('file_name')
        return col_names

    def normalize_df_by_feature(self, df):
        if self.univariate == True:
            start_del = 1
            end_del = -1
        else:
            start_del = 2
            end_del = -2
        df_max_scaled = df.copy()
        for i in range(len(self.feature_list)):
            i = i+1
            #get max of all rows with same feature
            if self.univariate:
                df_feature_max_scaled = df
            else:
                df_feature_max_scaled = df.loc[df['feature'] == float(i)]

            #get max value of all columns
            max = df_feature_max_scaled.iloc[:, start_del:end_del].abs().max().max()
            min = df_feature_max_scaled.iloc[:, start_del:end_del].abs().min().min()
            # apply normalization techniques
            for idx, row in df_max_scaled.iterrows():
                if self.univariate:
                    df_max_scaled.iloc[idx, start_del:end_del] = (df_max_scaled.iloc[idx, start_del:end_del].abs() - min)/ (max-min)
                else:
                    if row['feature'] == float(i):
                        df_max_scaled.iloc[idx, start_del:end_del] = (df_max_scaled.iloc[idx, start_del:end_del].abs() - min)/ (max-min)
        return df_max_scaled

    def normalize_df_by_window(self, df):
        if self.univariate == True:
            start_del = 1
            end_del = -1
        else:
            start_del = 2
            end_del = -2
        df_max_scaled = df.copy()

        df_feature_max_scaled = df
        for idx, row in df_max_scaled.iterrows():
            max = df_feature_max_scaled.iloc[idx, start_del:end_del].abs().max().max()
            min = df_feature_max_scaled.iloc[idx, start_del:end_del].abs().min().min()

            
            ##dirty fix for min-max issue when min = max
            divisor = max-min
            if divisor == 0:
                df_max_scaled.iloc[idx, start_del:end_del] = 0
            else:
                df_max_scaled.iloc[idx, start_del:end_del] = (df_max_scaled.iloc[idx, start_del:end_del].abs() - min)/ divisor
            
        return df_max_scaled

    def split_train_test(self, df, frac = 0.8, seed = 0):
        train = df.sample(frac=frac,random_state=seed)
        test = df.drop(train.index)
        return train, test

    def get_labeled_window(self, target, file_num, feature, window_list, file):
        t_arr = np.array([target])
        i_arr = np.array([file_num])
        i_f_arr = np.append(i_arr, feature+1)#index then feature(starting with 1)

        if self.univariate:
            labeled_arr = t_arr
        else:
            labeled_arr = i_f_arr
        for window in window_list:
            labeled_arr = np.append(labeled_arr, window)
        if not self.univariate:
            labeled_arr = np.append(labeled_arr, t_arr)
        labeled_arr = np.append(labeled_arr, [file[:-4]])#filename without csv
        return labeled_arr

    def stack_dataset(self, dataset_np, labeled_window):
        #check length of arr to make sure that files that are too short still can be used
        if len(labeled_window) == len(self.col_names):
            dataset_np = np.vstack([dataset_np, labeled_window])
        return dataset_np