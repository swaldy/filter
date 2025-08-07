# python modules
import sys
try:
    # os settings
    import os
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Standard library imports
    import math
    import h5py
    from fxpmath import Fxp
    import pandas as pd
    import csv
    import glob

    # Third-party imports
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf
    from pandas import read_csv
    from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    import hls4ml

    # TensorFlow imports
    from tensorflow.keras import datasets, layers, models
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
    from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from tensorflow.keras.metrics import SparseCategoricalAccuracy
    from tensorflow.keras.utils import Progbar
    from tensorflow.keras.models import clone_model

    # QKeras imports
    from qkeras import *

    # custom imports
    import utils as ut

except ImportError as e:
    print("Import error:", f"{__file__}: {str(e)}")
    sys.exit(1)  # Exit script immediately


# number with dense
def CreateModel(shape=16, nb_classes=3, first_dense=58, model_file=None):
    x = x_in = Input(shape, name="input")
    x = Dense(first_dense, name="dense1")(x)
    x = keras.layers.BatchNormalization()(x)
    x = Activation("relu", name="relu1")(x)
    x = Dense(nb_classes, name="dense2")(x)
    x = Activation("linear", name="linear")(x)
    model = Model(inputs=x_in, outputs=x)

    # load model file also
    if model_file:
        model = tf.keras.models.load_model(model_file)

    model.summary()
    return model

# Fold BatchNormalization in QDense
def CreateQModel(shape=16, model_file=None):
    x = x_in = Input(shape, name="input1")
    x = QDenseBatchnorm(58,
      kernel_quantizer=quantized_bits(4,0,alpha=1),
      bias_quantizer=quantized_bits(4,0,alpha=1),
      name="dense1")(x)
    x = QActivation("quantized_relu(8,0)", name="relu1")(x)
    x = QDense(3,
        kernel_quantizer=quantized_bits(4,0,alpha=1),
        bias_quantizer=quantized_bits(4,0,alpha=1),
        name="dense2")(x)
    x = Activation("linear", name="linear")(x)
    model = Model(inputs=x_in, outputs=x)

    # load model file also
    if model_file:
        co = {}
        utils._add_supported_quantized_objects(co)
        model = tf.keras.models.load_model(model_file, custom_objects=co)

    model.summary()
    return model

def custom_loss_function(y_true, y_pred):
    """
    Custom loss function.
    Default: Sparse Categorical Crossentropy with from_logits=True.
    Modify this function as needed.
    """

    # only valid y_true is 0,1,2. Any patterns with a 3 should be ignored 
    mask = tf.cast(tf.not_equal(y_true, 3), dtype=tf.float32) 

    # artificially set y_true = 3 values to 0
    y_true *= tf.cast(mask, dtype=tf.int64)

    # compute loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    # mask loss
    loss *= mask

    # compute mean loss, ignoring masked entries
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    return loss # tf.reduce_mean(loss)
    
class ModelPipeline:
    def __init__(self, model, optimizer, train_acc_metric, val_acc_metric, batch_size=32, asic_training=False):
        self.model = model
        self.optimizer = optimizer
        self.train_acc_metric = train_acc_metric
        self.val_acc_metric = val_acc_metric
        self.batch_size = batch_size
        self.asic_training = asic_training
        
        # Initialize custom loss function
        self.loss_fn = custom_loss_function

    def print_model(self):
        """
        Iterate through each layer and print weights and biases
        """
        for layer in self.model.layers:
            print(f"Layer: {layer.name}")
            for weight in layer.weights:
                print(f"  {weight.name}: shape={weight.shape}")
                print(f"    Values:\n{weight.numpy()}\n")

    def split_data(self, x_data, y_data, test_size=0.2, shuffle=True, random_state=42):
        """
        Splits data into training and testing datasets.
        """
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.batch_size)
    
    # @tf.function
    def forward_pass(self, x, training=False):
        """
        Performs a forward pass of the model.
        """
        return self.model(x, training=training)

    # @tf.function
    def train_step(self, 
                   x_batch, y_batch, 
                   dnn_csv = None, pixel_compout_csv = None, alpha = 0.1, # for asic training
    ):
        """
        Performs a single training step.
        """

        # training loop with gradients
        with tf.GradientTape() as tape:
            # evaluate model off
            # print("Forward pass")
            logits = self.forward_pass(x_batch, training=True)
            loss_value = self.loss_fn(y_batch, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        # print(grads)
        # print("Total Gradients: ", [g.numpy() for g in grads])
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y_batch, logits)
        return loss_value

    # @tf.function
    def val_step(self, x_batch, y_batch):
        """
        Performs a single validation step.
        """
        logits = self.forward_pass(x_batch, training=False)
        val_loss = self.loss_fn(y_batch, logits)
        self.val_acc_metric.update_state(y_batch, logits)
        return val_loss

    def train(self, epochs, patience=20):
        """
        Trains the model for a specified number of epochs with early stopping.
        """
        best_val_loss = float("inf")
        wait = 0
        best_weights = None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Training loop
            train_loss = 0
            train_steps = len(self.train_dataset)
            train_progbar = Progbar(target=train_steps, stateful_metrics=["loss", "sparse_categorical_accuracy"])
            
            for step, (x_batch, y_batch) in enumerate(self.train_dataset):
                # print(f"Batch {step}/{len(self.train_dataset)}")

                # compute loss value
                loss_value = self.train_step(x_batch, y_batch)
                
                # do training step
                train_loss += loss_value
                train_acc = self.train_acc_metric.result()
                train_progbar.update(step + 1, values=[("loss", loss_value.numpy()), ("sparse_categorical_accuracy", train_acc.numpy())])

            train_loss /= train_steps
            train_acc = self.train_acc_metric.result()
            self.train_acc_metric.reset_states()

            # Validation loop
            val_loss = 0
            val_steps = len(self.val_dataset)
            val_progbar = Progbar(target=val_steps, stateful_metrics=["val_loss", "val_sparse_categorical_accuracy"])
            
            for step, (x_batch, y_batch) in enumerate(self.val_dataset):
                val_loss_batch = self.val_step(x_batch, y_batch)
                val_loss += val_loss_batch
                val_acc = self.val_acc_metric.result()
                val_progbar.update(step + 1, values=[("val_loss", val_loss_batch.numpy()), ("val_sparse_categorical_accuracy", val_acc.numpy())])

            val_loss /= val_steps
            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                best_weights = self.model.get_weights()
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    self.model.set_weights(best_weights)
                    break