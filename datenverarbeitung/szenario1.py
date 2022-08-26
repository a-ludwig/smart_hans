import os
import numpy as np
import pandas as pd

path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export"

window_size = 40
df_len = 800

col_names =  ['index', 'class']

for i in range(window_size):
    col_names.append(i)

  
# create an empty dataframe
# with columns
dataset_df  = pd.DataFrame(columns = col_names)

file_num = 1
for file in os.listdir(path):

    print(file)
    start_annot = int(file.split("_")[5].split("-")[0])
    end_annot = int(file.split("_")[5].split("-")[1])

    anno_df_num = int(start_annot/window_size)
    
    df = pd.read_csv(path +"/"+ file, index_col=0)
    print(df.head)
    nosetip_df = df.iloc[:df_len,1]
    nosetip_df.columns = ['null']
    print(nosetip_df)

    
    for i in range(int(df_len/window_size)):
        temp_df = nosetip_df.iloc[i*window_size:(i+1)*window_size]
        temp_df = temp_df.rename(i * file_num)

        df = pd.DataFrame([[0 if (i < anno_df_num) else 1]], columns=['zeile'])
        temp_df = pd.concat([df, temp_df])
        #temp_df = temp_df.rename(col'unbena', inplace = True)
        dataset_df = pd.concat([dataset_df, temp_df], axis = 0, ignore_index=False)
    print('df')
    print(temp_df)
    print(dataset_df.head)

    file_num = file_num + 1
    break