# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:21:40 2018

@author: mljmo
"""
import os
import pandas as pd

Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data\joinedtracks"
os.chdir(Path)
Data = []
for file in os.listdir():
    Data.append(pd.read_csv(os.path.join(Path,file)))
    os.remove(file)
    
df = pd.concat(Data)
df.to_csv("joinedTracks.csv")