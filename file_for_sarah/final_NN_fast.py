import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

sizes = ['50x10', '50x12P5', '50x15', '50x20', '50x25', '100x25', '100x25x150']
thresholds = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
prime_num = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

dataset_name = '/uscms/home/swaldych/nobackup/dataset_3s_400NoiseThresh'
results_dir = os.path.join(dataset_name, 'results')
models_dir = os.path.join(dataset_name, 'models')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def thresh_tag(threshold):
    return '0P' + str(threshold - int(threshold))[2:]

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(14,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

for sensor_geom in sizes:
    for threshold in thresholds:
        tag = thresh_tag(threshold)

        print("=" * 60)
        print(f"Loading data once for {sensor_geom}, threshold={threshold}")

        X_train = pd.read_csv(
            f'{dataset_name}/FullPrecisionInputTrainSet_{sensor_geom}_{tag}thresh.csv'
        ).to_numpy()

        y_train = pd.read_csv(
            f'{dataset_name}/TrainSetLabel_{sensor_geom}_{tag}thresh.csv'
        ).to_numpy().ravel()

        X_test = pd.read_csv(
            f'{dataset_name}/FullPrecisionInputTestSet_{sensor_geom}_{tag}thresh.csv'
        ).to_numpy()

        y_test = pd.read_csv(
            f'{dataset_name}/TestSetLabel_{sensor_geom}_{tag}thresh.csv'
        ).to_numpy().ravel()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for run_iter in range(10):
            print("=" * 30)
            print(f"Run {run_iter}: Training model for {sensor_geom} at pT boundary = {threshold}")

            tf.keras.backend.clear_session()
            tf.random.set_seed(prime_num[run_iter])

            model = build_model()

            es = EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                mode='max',
                patience=20,
                restore_best_weights=True
            )

            history = model.fit(
                X_train,
                y_train,
                callbacks=[es],
                epochs=200,
                batch_size=1024,
                validation_split=0.2,
                shuffle=True,
                verbose=0
            )

            epochs = range(1, len(history.history['loss']) + 1)

            plt.figure()
            plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
            plt.plot(epochs, history.history['val_loss'], 'orange', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{results_dir}/loss_{sensor_geom}_{tag}thresh_run{run_iter}.png')
            plt.close()

            plt.figure()
            plt.plot(epochs, history.history['sparse_categorical_accuracy'], 'bo', label='Training accuracy')
            plt.plot(epochs, history.history['val_sparse_categorical_accuracy'], 'orange', label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f'{results_dir}/accuracy_{sensor_geom}_{tag}thresh_run{run_iter}.png')
            plt.close()

            preds = model.predict(X_test, verbose=0)
            predictionsFiles = np.argmax(preds, axis=1)

            pd.DataFrame({'predict': predictionsFiles}).to_csv(
                f'{results_dir}/predictionsFiles_{sensor_geom}_{tag}thresh_run{run_iter}.csv',
                index=False
            )

            pd.DataFrame({'true': y_test}).to_csv(
                f'{results_dir}/testResults_{sensor_geom}_{tag}thresh_run{run_iter}.csv',
                index=False
            )

            plt.figure()
            plt.hist(y_test, bins=30)
            plt.savefig(f'{results_dir}/ytest_hist_{sensor_geom}_{tag}thresh_run{run_iter}.png')
            plt.close()

            score = model.evaluate(X_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

            disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictionsFiles)
            disp.figure_.suptitle("Multiclassifier Confusion Matrix")
            plt.savefig(f'{results_dir}/confusionMatrix_{sensor_geom}_{tag}_run{run_iter}.png')
            plt.close()
            print(f"Confusion matrix:\n{disp.confusion_matrix}")
            model.save_weights(f'{models_dir}/trained_model_{sensor_geom}_{tag}_run{run_iter}.weights.h5')
            model.save(f'{models_dir}/trained_model_{sensor_geom}_{tag}_run{run_iter}.h5')

            del model, history, preds, predictionsFiles, disp, score
            gc.collect()
            tf.keras.backend.clear_session()

        del X_train, y_train, X_test, y_test, scaler
        gc.collect()