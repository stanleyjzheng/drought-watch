import argparse
import os
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers, backend
!pip install -q efficientnet
import efficientnet.tfkeras as efn
from glob import glob

AUTO = tf.data.experimental.AUTOTUNE
batch_size=64

# Deotte augmentations
ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0

def load_data(data_path):
  train = file_list_from_folder("train", data_path)
  test = file_list_from_folder("val", data_path)
  return train, test


keylist = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']

def read_labeled_tfrecord(example):
    tfrec_format = {
      'B1': tf.io.FixedLenFeature([], tf.string),
      'B2': tf.io.FixedLenFeature([], tf.string),
      'B3': tf.io.FixedLenFeature([], tf.string),
      'B4': tf.io.FixedLenFeature([], tf.string),
      'B5': tf.io.FixedLenFeature([], tf.string),
      'B6': tf.io.FixedLenFeature([], tf.string),
      'B7': tf.io.FixedLenFeature([], tf.string),
      'B8': tf.io.FixedLenFeature([], tf.string),
      'B9': tf.io.FixedLenFeature([], tf.string),
      'B10': tf.io.FixedLenFeature([], tf.string),
      'B11': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.int64),
    }              
    example = tf.io.parse_single_example(example, tfrec_format)
    return [example[i] for i in keylist], tf.cast(example['label'], tf.uint8) # required float32 for tpu, CHANGE CLIP

def read_unlabeled_tfrecord(example):
    tfrec_format = {
      'B1': tf.io.FixedLenFeature([], tf.string),
      'B2': tf.io.FixedLenFeature([], tf.string),
      'B3': tf.io.FixedLenFeature([], tf.string),
      'B4': tf.io.FixedLenFeature([], tf.string),
      'B5': tf.io.FixedLenFeature([], tf.string),
      'B6': tf.io.FixedLenFeature([], tf.string),
      'B7': tf.io.FixedLenFeature([], tf.string),
      'B8': tf.io.FixedLenFeature([], tf.string),
      'B9': tf.io.FixedLenFeature([], tf.string),
      'B10': tf.io.FixedLenFeature([], tf.string),
      'B11': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.int64),
    }              
    example = tf.io.parse_single_example(example, tfrec_format)
    return [example[i] for i in keylist] # required float32 for tpu, CHANGE CLIP

def prepare_image(img, augment=True, dim=256):    
    img = tf.io.decode_raw(img, out_type=tf.uint8)
    img = tf.cast(img, tf.float32) / 255.0 # CHEC
    
    if augment:
        img = transform(img,DIM=dim)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
            
    return tf.cast(tf.reshape(img, [dim,dim,10]), tf.float32)

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def get_dataset(files, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_names=True, batch_size=16, dim=256):
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle: 
        ds = ds.shuffle(batch_size)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else: 
        ds = ds.map(read_unlabelled_tfrecord, num_parallel_calls=AUTO)
    
    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, dim=dim), tf.one_hot(imgname_or_label, 4, dtype=tf.float32), [4]), num_parallel_calls=AUTO)
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return ds

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

p_min, p_max = 0.005, 0.99
def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,p_min,p_max)
    return -backend.mean(y_true*backend.log(y_pred) + (1-y_true)*backend.log(1-y_pred))

def build_model(dim=256, ef=0):
    inp = tf.keras.layers.Input(shape=(dim,dim,10))
    base = EFNS[ef](input_shape=(dim,dim,10),weights=None,include_top=False) #Change imagnet to noisy-student here
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=0)
    model.compile(optimizer=opt,loss='mse', metrics=[logloss, 'mse', 'acc'])
    return model

def find_files(path):
    test = glob(f"{path}/val/*")
    train = glob(f"{path}/train/*")
    return train,test

trainpath, testpath = find_files('../input/drought-watch/droughtwatch_data')
train = get_dataset(trainpath, dim=65)
test = get_dataset(testpath, dim=65)

ckp = tf.keras.callbacks.ModelCheckpoint(f'droughtwatch.hdf5', monitor = 'val_logloss', verbose = 0, save_best_only = True, save_weights_only = True, mode = 'min')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, mode='min', min_lr=1e-5, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=10, mode='min',restore_best_weights=True)

model = build_model(dim=65, ef=0)
model.fit(train, validation_data = test, epochs=15, batch_size=batch_size, callbacks=[reduce_lr, early_stopping, ckp])