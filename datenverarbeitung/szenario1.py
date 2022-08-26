import os
import numpy as np
import pandas as pd

path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export"


for file in os.listdir(path):

    print(file)
    start_annot = int(file.split("_")[5].split("-")[0])
    end_annot = int(file.split("_")[5].split("-")[1])
    
    df = pd.read_csv(path +"/"+ file, index_col=0)
    print(df.head)
    nosetip_df = df.iloc[:,1]
    print(nosetip_df)

    window_size = 40
    dataset = nosetip_df.iloc[0:window_size].T



    print(dataset.head)
    break