from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import math
from tensorflow.keras.optimizers import Adam

size = '12P5'
threshold = 0.5
sensor_geom = '50x'+size
df1 = pd.read_csv('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/FullPrecisionInputTrainSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
df1

df2 = pd.read_csv('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/TrainSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
df2

df3 = pd.read_csv('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/FullPrecisionInputTestSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
df3

df4 = pd.read_csv('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/TestSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
df4

X_train = df1.values
X_test = df3.values

y_train = df2.values
y_test = df4.values
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

X_test

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(14,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # default from_logits=False
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

model.summary()

es = EarlyStopping(monitor='val_sparse_categorical_accuracy', 
                                   mode='max', # don't minimize the accuracy!
                                   patience=20,
                                   restore_best_weights=True)

history = model.fit(X_train,
                    y_train,
                    callbacks=[es],
                    epochs=200, 
                    batch_size=1024,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=1)

history_dict = history.history
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss'] 
epochs = range(1, len(loss_values) + 1) 
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/results/loss_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.png')

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#np.max(val_acc)
plt.savefig('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/results/accuracy_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.png')


preds = model.predict(X_test) 
predictionsFiles =np.argmax(preds, axis=1)

pd.DataFrame(predictionsFiles).to_csv("/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/results/predictionsFiles_"+sensor_geom+"_0P"+str(threshold - int(threshold))[2:]+"thresh.csv",header='predict', index=False)

pd.DataFrame(y_test).to_csv("/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/results/testResults_"+sensor_geom+"_0P"+str(threshold - int(threshold))[2:]+"thresh.csv",header='true', index=False)
plt.hist(y_test, bins=30)
plt.savefig('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/results/hist_of_y_test'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.png')


score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

from sklearn import datasets, svm, metrics
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictionsFiles)
disp.figure_.suptitle("Multiclassifier Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.savefig('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/results/confusionMatrix_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'.png')


model.save_weights('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/models/trained_model_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:])

model.save('/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh/models/trained_model_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'.h5')
