import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import os

#from HANS_Repo.Machine_learning.datenverarbeitung.dataloader import dataloader

class visualizer:
    def __init__(self, dl, path = None, show = False):
        self.df = dl.df_labled
        self.dl = dl
        self.show = show
        self.path = path
        self.colors = ['r', 'g', 'b', 'c']
        if path == None:
            self.show = True
        
        if 'feature' in self.df.columns:
            self.univariate = False
        else:
            self.univariate = True
    def visualize(self, fname):
        df_len = self.df.shape[0]
        num_col = 4
        


        if self.dl.univariate:
            num_rows = 3
            plots_per_sub = int(df_len / (num_col*num_rows))
            
        else:
            num_f = len(self.dl.feature_list)
            num_rows = int(num_f/num_col)+1
            plots_per_sub = num_rows * num_col
        
        num_target = int(self.df['target'].abs().max())
        for index, row in self.df.iterrows():
            target = int(row['target'])
            feature = int(row['feature'])
            
            file_name = row['file_name']
            file_info = file_name.split("_")

            timestamp = file_info[2] + '_' + file_info[3]
            ax = plt.subplot(num_rows, num_col, feature)
            

            temp_df = self.df.iloc[index, 2:-2]

            ax.set_title(self.dl.feature_list[feature-1])
            L = temp_df.plot( ax=ax, color = self.colors[target], sharex = True, sharey = True, figsize = (10,8), label = f'class {target}')
            pos = temp_df.size-3
            plt.text(x = float(pos), y = temp_df.iloc[pos].item(), s= timestamp)

        self.legend_without_duplicate_labels(ax)

        self.show()
        return

    def visualize_raw(self, path):
        df_len = 800
        y = list(range(0,df_len))
        
        for file in os.listdir(path):
            fig = plt.figure(figsize = (10,8))
            print(file)
            start_annot = int(file.split("_")[5].split("-")[0]) 
            end_annot = int(file.split("_")[5].split("-")[1])
            target_tap_nr = int(start_annot/40)

            file_arr = np.genfromtxt(path + '/' + file, skip_header=True, delimiter=',')
            #kürzen des arrays
            file_arr = file_arr[:df_len]
            file_arr_0 = file_arr[:start_annot]
            y_0 = y[:start_annot]
            file_arr_1 = file_arr[start_annot:end_annot]
            y_1 = y[start_annot:end_annot]
            file_arr_2 = file_arr[end_annot:]
            y_2 = y[end_annot:]

            for i, column in enumerate(self.dl.column_dict):
                ax = fig.add_subplot(5, 5, i+1)
                temp_arr = file_arr_0[:,i+1]
                ax.plot(y_0,temp_arr, color = self.colors[0])
                ax.plot(y_1,file_arr_1[:,i+1], color = self.colors[1])
                ax.plot(y_2, file_arr_2[:,i+1], color = self.colors[2])
                ax.set_title(list(self.dl.column_dict.keys())[list(self.dl.column_dict.values()).index(i+1)])
            #break
            fig.suptitle(file)
            plt.show()
        

    def legend_without_duplicate_labels(self, ax): # https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib/56253636#56253636
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique),loc='upper left')

    def show(self, fname):
        if self.show:
            plt.show()
        else:
            plt.savefig(fname= self.path + fname, format = 'PNG', dpi = 'figure')
        return