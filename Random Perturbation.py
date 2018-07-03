# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:45:12 2018

@author: mljmo
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def perturbationMask(df, std):
    xMin, xMax, yMin, yMax = df["X"].min(), df["X"].max(), df["Y"].min(), df["Y"].max()
    xstd = std * (xMax-xMin)
    ystd = std * (yMax-yMin)
    df["X"] = df["X"].apply(lambda x: x+np.random.normal(0, xstd))
    df["Y"] = df["Y"].apply(lambda y: y+np.random.normal(0, ystd))
    return df


Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data\joinedtracks"
os.chdir(Path)    
df = pd.read_csv("joinedTracks.csv")

track = df.head(500)
df = track.copy()
df = perturbationMask(df, 0.1)



plt.subplot(2,1,1)
plt.title("Original data")
plt.scatter(track["X"], track["Y"])
plt.subplot(2,1,2)
plt.title("applied grid masking")
plt.scatter(df["X"], df["Y"],  c='red', marker=',')
track==df