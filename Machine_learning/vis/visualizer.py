import matplotlib.pyplot as plt
import pandas as pd
import random

#from HANS_Repo.Machine_learning.datenverarbeitung.dataloader import dataloader

class visualizer:
    def __init__(self, dl, path = None, show = False):
        self.df = dl.df_labled
        self.dl = dl
        self.show = show
        self.path = path
        if path == None:
            self.show = True
        
        if 'feature' in self.df.columns:
            self.univariate = False
        else:
            self.univariate = True
    def visualize(self, fname):
        df_len = self.df.shape[0]
        num_col = 4
        colors = ['r', 'g', 'b', 'c']


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
            L = temp_df.plot( ax=ax, color = colors[target], sharex = True, sharey = True, figsize = (10,8), label = f'class {target}')
            pos = temp_df.size-3
            plt.text(x = float(pos), y = temp_df.iloc[pos].item(), s= timestamp)

        self.legend_without_duplicate_labels(ax)


            

        if self.show:
            plt.show()
        else:
            plt.savefig(fname= self.path + fname, format = 'PNG', dpi = 'figure')
        return

    def legend_without_duplicate_labels(self, ax): # https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib/56253636#56253636
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique),loc='upper left')