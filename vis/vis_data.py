import matplotlib.pyplot as plt
import pandas as pd
import random

def visualize(df):
    x,y = df.shape
    plots_per_sub = int(x / 12)
    colors = ['r', 'g', 'b']
    for i in range(12):
        i = i + 1
        for j in range(plots_per_sub):
            ax = plt.subplot(3, 4, i)

            row = j*(i)

            target = int(df.iloc[row, 0].item())
            temp_df = df.iloc[row, 1:]
            temp_df.plot(ax=ax, color = colors[target])
    plt.show()
    return