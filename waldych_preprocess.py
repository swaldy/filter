import numpy as np
import pandas as pd
from pandas import read_csv
import math
import matplotlib.pyplot as plt
import glob

# Global variables
threshold = 0.5
noise_threshold = 400
sensor_geom = '100x25x150_1100fb'
dataset_savedir =  '/eos/user/s/swaldych/smart_pix/labels/preprocess' # for save loc of final datasets

dirtrain = '/eos/user/s/swaldych/smart_pix/labels/'
# /location/of/parquets/smartpixels/dataset_2s/dataset_2s_50x12P5_parquets/unflipped
dftrain = pd.read_parquet(dirtrain+'labels_d16401.parquet')
print(dftrain.head())
print(dftrain.tail())

trainlabels = []
trainrecons = []

iter=0
suffix = 16400
for filepath in glob.iglob(dirtrain+'labels*.parquet'):
    iter+=3
print(iter," files present in directory.")
for i in range(int(iter/3)):
        trainlabels.append(pd.read_parquet(dirtrain+'labels_d'+str(suffix+i+1)+'.parquet'))
        #trainrecons.append(pd.read_parquet(dirtrain+'recon2D_d'+str(suffix+i+1)+'.parquet'))
trainlabels_csv = pd.concat(trainlabels, ignore_index=True)
#trainrecons_csv = pd.concat(trainrecons, ignore_index=True)

iter_0, iter_1, iter_2 = 0, 0, 0
iter_rem = 0
for iter, row in trainlabels_csv.iterrows():
    if(abs(row['pt'])>threshold):
        iter_0+=1
    elif(-1*threshold<=row['pt']<0):
        iter_1+=1
    elif(0<row['pt']<=threshold):
        iter_2+=1
    else:
        iter_rem+=1
print("iter_0: ",iter_0)
print("iter_1: ",iter_1)
print("iter_2: ",iter_2)
print("iter_rem: ",iter_rem)

plt.hist(trainlabels_csv['pt'], bins=100)
plt.title('pT of all events')
plt.show()

plt.hist(trainlabels_csv[abs(trainlabels_csv['pt'])>threshold]['pt'], bins=100)
plt.title('pT of Class 0 events')
plt.show()

plt.hist(trainlabels_csv[(0<=trainlabels_csv['pt'])&(trainlabels_csv['pt']<=threshold)]['pt'], bins=50)
plt.hist(trainlabels_csv[(-1*threshold<=trainlabels_csv['pt'])& (trainlabels_csv['pt']<0)]['pt'], bins=50)
plt.title('pT of Class 1+2 events')
plt.show()

number_of_events = (min(iter_1, iter_2)//1000)*1000
if(number_of_events*2>iter_0):
    number_of_events = (iter_0//1000)*1000/2
number_of_events = int(number_of_events)
print("Number of events: ",number_of_events)

import sys
np.set_printoptions(threshold=sys.maxsize)
def sumRow(X):
    X = np.where(X < noise_threshold, 0, X)
    sum1 = 0
    sumList = []
    for i in X:
        sum1 = np.sum(i,axis=0)
        sumList.append(sum1)
        b = np.array(sumList)
    return b

# --- build class from pt only, no recon2D ---
df = trainlabels_csv.copy()

df["cls"] = -1
df.loc[df["pt"].abs() > threshold, "cls"] = 0
df.loc[(df["pt"] >= -threshold) & (df["pt"] < 0), "cls"] = 1
df.loc[(df["pt"] >= 0) & (df["pt"] <= threshold), "cls"] = 2

traindf_all = df[["y-local", "cls", "pt"]].copy()
train = train.drop(['cls', 'pt'], axis=1)

train.to_csv(dataset_savedir+'/FullPrecisionInputTrainSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
trainlabel.to_csv(dataset_savedir+'/TrainSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
trainpt.to_csv(dataset_savedir+'/TrainSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
