import os
import pandas as pd

path = '/Users/adi/Documents/code/tsai_test/single_file_test/'
data = []
window_size = 40


for file in os.listdir(path) :
    data = []
    print(path+file)
    full_path = path+file
    input_data = pd.read_csv(full_path, sep=",")
    iterator = 0
    initial_iterator = iterator
    start_annot = int(file.split("_")[5].split("-")[0])
    end_annot = int(file.split("_")[5].split("-")[1])
    for index, row in input_data.iterrows():
        while iterator <= (initial_iterator+window_size) :
                print(row['nosetip_y'])
                if index < start_annot:
                    annotate = 0
                if index >= start_annot and index <= end_annot :
                            #print(" i will annotate at frame "+ str(current_frame))
                    annotate = 1
                if index > end_annot :
                    annotate = 2
                data.append( {
                            iterator : row['nosetip_y'],
                            "target" : annotate
                        } )
                print(iterator)
                iterator = iterator + 1
        print (data)
        iterator = 0
df2 = pd.DataFrame(data)
df2
