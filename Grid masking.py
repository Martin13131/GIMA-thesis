# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:44:58 2018

@author: mljmol
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data\joinedtracks"
os.chdir(Path)

def formGrid(df, gridPoints=10):
    xMin, xMax, yMin, yMax = df["X"].min(), df["X"].max(), df["Y"].min(), df["Y"].max()
    gridX = np.linspace(xMin, xMax, gridPoints)
    gridY = np.linspace(yMin, yMax, gridPoints)
    return gridX, gridY

def findClosest(val, gridVals): # Use scipy spatial.euclidean?
    idx = np.argmin(abs(val-gridVals))
    return gridVals[idx]
    
def snapToGrid(df, gridX, gridY):
    X = df["X"].apply(lambda x: findClosest(x, gridX))
    Y = df["Y"].apply(lambda y: findClosest(y, gridY))
    df["X"] = X
    df["Y"] = Y
    return df


def gridMask(df, gridPoints=10):
    gridX, gridY = formGrid(df, gridPoints)
    df = snapToGrid(df, gridX, gridY)
    return df
    

df = pd.read_csv("joinedTracks.csv")

track = df.head(500)
df = track.copy()
df = gridMask(df, gridPoints=10)

gridX, gridY = formGrid(df, gridPoints=10)
xplot, yplot = [], []
for x in gridX:
    for y in gridY:
        xplot.append(x)
        yplot.append(y)

plt.scatter(xplot, yplot, c='purple', marker=',')

plt.subplot(2,1,1)
plt.title("Original data")
plt.scatter(track["X"], track["Y"], c='b', marker=',')
plt.subplot(2,1,2)
plt.title("applied grid masking")
plt.scatter(df["X"], df["Y"], c='red', marker=',')
track==df




#plt.scatter(track["X"], track["Y"])
#[track["X"], track["Y"]] == [df["X"],df["Y"]]
