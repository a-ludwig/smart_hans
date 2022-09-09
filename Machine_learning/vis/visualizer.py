import matplotlib.pyplot as plt
import pandas as pd
import random

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
        plots_per_sub = int(df_len / 12)
        colors = ['r', 'g', 'b', 'c']
        for i in range(12):
            i = i + 1
            for j in range(plots_per_sub):
                ax = plt.subplot(3, 4, i)
                
                row = j*(i)
                if self.univariate:
                    target = int(self.df.iloc[row, 0].item())
                    temp_df = self.df.iloc[row, 1:-1]
                else:
                    target = int(self.df.iloc[row, -2].item())
                    temp_df = self.df.iloc[row, 2:-2]

                temp_df.plot(ax=ax, color = colors[target], sharex = True, sharey = True, figsize = (10,8))

        if self.show:
            plt.show()
        else:
            plt.savefig(fname= self.path + fname, format = 'PNG', dpi = 'figure')
        return