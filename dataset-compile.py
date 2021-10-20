"""
Created on Wed Oct 20 12:55:42 2021

@author: cburgheimer
ONLY RUN IF YOU WANT A COMPILED DATASET LOCALLy, IT IS OVER A MILLION SENTENCES TOTAL
"""
import os
import pandas as pd

csv_filename = 'data/data_large.csv'
datadir = 'data'
dataframes = []
for txt in os.listdir(datadir):
    path = os.path.join(datadir, txt)
    if os.path.isfile(path):
        dataframe = pd.DataFrame.read_csv(path)
        dataframes.append(dataframe)

df = pd.concat(dataframes)
df.to_csv(csv_filename, encoding = 'utf-8')
