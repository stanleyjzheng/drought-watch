import argparse
import os
import math
import tensorflow as tf
from tensorflow.keras import layers, initializers, backend
from glob import glob
import numpy as np

AUTO = tf.data.experimental.AUTOTUNE

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
    return [example[i] for i in keylist], tf.cast(example['label'], tf.float32) # required float32 for tpu, CHANGE CLIP

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
    
    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, dim=dim), tf.reshape(imgname_or_label, [1])), num_parallel_calls=AUTO)
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return ds

if __name__=="__main__":
    dataset = get_dataset(['input/drought-watch/droughtwatch_data/val/part-r-00000'], dim=65)
    for x, y in dataset:
        # print(np.array(x[0]).shape, np.array(y[0]).shape)
        for num,(x_1, y_1) in enumerate(zip(x, y)):
            np.save(f"example_images/label_{int(y_1[0])}_image_num_{num+1}", np.array(x_1))
            break
        break