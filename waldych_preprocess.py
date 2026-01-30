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

dirtrain = '/eos/user/s/swaldych/smart_pix/labels'
# /location/of/parquets/smartpixels/dataset_2s/dataset_2s_50x12P5_parquets/unflipped
dftrain = pd.read_parquet(dirtrain+'labels_d16401.parquet')
print(dftrain.head())
print(dftrain.tail())
