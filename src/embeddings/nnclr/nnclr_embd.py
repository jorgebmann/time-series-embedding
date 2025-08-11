import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
import tensorflow as tf
from tabulate import tabulate # for verbose tables
from tensorflow.keras.utils import to_categorical # for one-hot encoding
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
#from data_loader_n import load_unimib_dataset
from scripts.augmentations import augmenter
# from encoder import encoder
from scripts.nnclr import NNCLR

def get_feature_vectors(model, dataset):

  feature_vectors = model.encoder(dataset, training=False)
  np_data=feature_vectors.numpy()
  feature_vectors_df = pd.DataFrame(np_data)
  return feature_vectors_df

def nnclr_lstm_embedding(x_train_df,x_val_df,x_test_df,y_train_df, y_val_df, y_test_df, width, n_classes):
    # Add an extra dimension to the x datasets
    train_sc= np.expand_dims(x_train_df, axis=-1)
    val_sc = np.expand_dims(x_val_df, axis=-1)
    test_sc= np.expand_dims(x_test_df, axis=-1)

    # Reshape the y datasets to remove the singleton dimension
    y_train_df = y_train_df.values.reshape(-1)
    y_val_df = y_val_df.values.reshape(-1)
    y_test_df = y_test_df.values.reshape(-1)

    # Print the new shapes to verify
    print("x_train_df shape:", x_train_df.shape)
    print("y_train_df shape:", y_train_df.shape)
    print("x_val_df shape:", x_val_df.shape)
    print("y_val_df shape:", y_val_df.shape)
    print("x_test_df shape:", x_test_df.shape)
    print("y_test_df shape:", y_test_df.shape)
    input_shape = train_sc[0].shape
    # Constants
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    AUTOTUNE = tf.data.AUTOTUNE
    temperature = 0.1        # the temperature for the softmax function in the contrastive loss
    queue_size = 5000        # the size of the queue for storing the feature vectors of the neighbors

    pretrain_num_epochs = 20   # the number of epochs to pretrain the model
    finetune_num_epochs = 20   # The number of epochs to fine-tune the model.

    # Create the encoder model
    encoder_model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.LSTM(128, return_sequences=True, activation='tanh'),
            layers.Dropout(0.2),  # Add dropout after the first LSTM layer
            layers.LSTM(128, return_sequences=True, activation='tanh'),  # Retain sequences
            layers.Dropout(0.5),
            layers.GlobalMaxPooling1D(),  # Use GlobalMaxPooling1D instead of Flatten
            layers.Dense(width, activation='relu')  # Adjust the width as necessary
        ],
        name="encoder_model",
    )
    print(encoder_model.summary())
    model = NNCLR(temperature=temperature, queue_size=queue_size, input_shape=input_shape, output_width=width, n_classes= n_classes, encoder=encoder_model)

    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
        run_eagerly=None # True = run eagerly, False = run as graph, None = autodetect
    )

    model.build(input_shape=(None, input_shape[0], input_shape[1]))
    print(model.summary())
    unlabeled_train_dataset = tf.data.Dataset.from_tensor_slices((train_sc, y_train_df))
    unlabeled_train_dataset = unlabeled_train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    unlabeled_train_dataset = unlabeled_train_dataset.prefetch(AUTOTUNE)

    labeled_train_dataset = tf.data.Dataset.from_tensor_slices((val_sc, y_val_df))
    labeled_train_dataset = labeled_train_dataset.batch(BATCH_SIZE)
    labeled_train_dataset = labeled_train_dataset.prefetch(AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_sc, y_test_df))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    train_dataset = tf.data.Dataset.zip((unlabeled_train_dataset, labeled_train_dataset)).prefetch(buffer_size=AUTOTUNE)
    pretrain_history = model.fit(
    train_dataset, epochs=pretrain_num_epochs, validation_data=test_dataset,
    verbose=2 # 0 = silent, 1 = progress bar, 2 = one line per epoch, 3 = one line per batch
              # Due to a weird bug, the fit function crashes if verbose is set to 1.
        )
    train_emb=get_feature_vectors(model, train_sc)
    test_emb=get_feature_vectors(model, test_sc)
    val_emb=get_feature_vectors(model, val_sc)

    return train_emb,val_emb,test_emb

def nnclr_cnn_embedding(x_train_df,x_val_df,x_test_df,y_train_df,  y_val_df, y_test_df,width, n_classes):
    # Add an extra dimension to the x datasets
    train_sc= np.expand_dims(x_train_df, axis=-1)
    val_sc = np.expand_dims(x_val_df, axis=-1)
    test_sc= np.expand_dims(x_test_df, axis=-1)

    # Reshape the y datasets to remove the singleton dimension
    y_train_df = y_train_df.values.reshape(-1)
    y_val_df = y_val_df.values.reshape(-1)
    y_test_df = y_test_df.values.reshape(-1)

    # Print the new shapes to verify
    print("x_train_df shape:", x_train_df.shape)
    print("y_train_df shape:", y_train_df.shape)
    print("x_val_df shape:", x_val_df.shape)
    print("y_val_df shape:", y_val_df.shape)
    print("x_test_df shape:", x_test_df.shape)
    print("y_test_df shape:", y_test_df.shape)
    input_shape = train_sc[0].shape
    # Constants
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    AUTOTUNE = tf.data.AUTOTUNE
    temperature = 0.1        # the temperature for the softmax function in the contrastive loss
    queue_size = 5000        # the size of the queue for storing the feature vectors of the neighbors

    pretrain_num_epochs = 20   # the number of epochs to pretrain the model
    finetune_num_epochs = 20   # The number of epochs to fine-tune the model.

    # Create the encoder model
    encoder_model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv1D(filters=100, kernel_size=10, activation='relu'),
                layers.Conv1D(filters=100, kernel_size=10, activation='relu'),
                layers.Dropout(0.5),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(width, activation='relu')
            ],
            name="encoder_model",
        )
    print(encoder_model.summary())
    model = NNCLR(temperature=temperature, queue_size=queue_size, input_shape=input_shape, output_width=width, n_classes= n_classes, encoder=encoder_model)

    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
        run_eagerly=None # True = run eagerly, False = run as graph, None = autodetect
    )

    model.build(input_shape=(None, input_shape[0], input_shape[1]))
    print(model.summary())
    unlabeled_train_dataset = tf.data.Dataset.from_tensor_slices((train_sc, y_train_df))
    unlabeled_train_dataset = unlabeled_train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    unlabeled_train_dataset = unlabeled_train_dataset.prefetch(AUTOTUNE)

    labeled_train_dataset = tf.data.Dataset.from_tensor_slices((val_sc, y_val_df))
    labeled_train_dataset = labeled_train_dataset.batch(BATCH_SIZE)
    labeled_train_dataset = labeled_train_dataset.prefetch(AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_sc, y_test_df))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    train_dataset = tf.data.Dataset.zip((unlabeled_train_dataset, labeled_train_dataset)).prefetch(buffer_size=AUTOTUNE)
    pretrain_history = model.fit(
    train_dataset, epochs=pretrain_num_epochs, validation_data=test_dataset,
    verbose=2 # 0 = silent, 1 = progress bar, 2 = one line per epoch, 3 = one line per batch
              # Due to a weird bug, the fit function crashes if verbose is set to 1.
    )
    train_emb=get_feature_vectors(model, train_sc)
    test_emb=get_feature_vectors(model, test_sc)
    val_emb=get_feature_vectors(model, val_sc)

    return train_emb,val_emb,test_emb
