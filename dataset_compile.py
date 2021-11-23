"""
Created on Wed Oct 20 12:55:42 2021

@author: cburgheimer
ONLY RUN IF YOU WANT A COMPILED DATASET LOCALLy, IT IS OVER A MILLION SENTENCES TOTAL
"""
import os
import pandas as pd

csv_filename = 'data_with_labels/data_sampled.csv'
datadir = 'data'
dataframes = []
for txt in os.listdir(datadir):
    path = os.path.join(datadir, txt)
    if os.path.isfile(path):
        dataframe = pd.read_csv(path)
        dataframes.append(dataframe)

df = pd.concat(dataframes)
df_new = df.sample(7000, random_state=42)
df_new.to_csv(csv_filename,header=False,index=False)
