%env TF_USE_LEGACY_KERAS 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, svm, metrics
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
# import keras
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import math
from tensorflow.keras.optimizers import Adam

sensor_geom = "100x25x150_1100fb"
threshold = 0.1 #in GeV
seed = 13  

tf.random.set_seed(seed)

tag = f"{sensor_geom}_0P{str(threshold - int(threshold))[2:]}thresh"

print("=============================")
print(f"Training model for {sensor_geom} at pT boundary = {threshold}, seed={seed}")

df1 = pd.read_csv(f"./{dataset_name}/FullPrecisionInputTrainSet_{tag}.csv")
df2 = pd.read_csv(f"./{dataset_name}/TrainSetLabel_{tag}.csv")
df3 = pd.read_csv(f"./{dataset_name}/FullPrecisionInputTestSet_{tag}.csv")
df4 = pd.read_csv(f"./{dataset_name}/TestSetLabel_{tag}.csv")

X_train = df1.values
X_test  = df3.values
y_train = df2.values.ravel()  # <-- important: make it 1D for SparseCategoricalCrossentropy
y_test  = df4.values.ravel()

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# IMPORTANT: your new dataset is 1 feature (y-local), not 14.
input_dim = X_train.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(
    optimizer=Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

es = EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    callbacks=[es],
    epochs=200,
    batch_size=1024,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)
