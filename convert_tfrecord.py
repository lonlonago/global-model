# -*- coding:utf-8 -*-
# read and convert ai challenger label json file


import numpy as np
import tensorflow as tf
import datetime
import math
import sys
import random
import threading
import json
import os
import matplotlib.pyplot as plt
import shutil
from prepare_data_global_model import prepare_global_label



def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id, filename):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(filename),
  }))



# 图片读取类，有两个方法，分别可以读图片的维度，返回图片的宽和高、 对图片原始数据进行转码
# 但是它实际上就起到了一个作用，获取图片的尺寸
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


# 获取要输出的tfrecord的文件名称，最后格式类似于 flowers_train_00001-of-00005.tfrecord
def _get_dataset_filename(TFRECORD_TARGET_PATH, split_name, shard_id, _NUM_SHARDS):
  output_filename = '%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(TFRECORD_TARGET_PATH, output_filename)



def _convert_dataset( split_name, filenames, labels, TFRECORD_TARGET_PATH, _NUM_SHARDS):
  # """Converts the given filenames to a TFRecord dataset.

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(TFRECORD_TARGET_PATH, split_name, shard_id,_NUM_SHARDS)

        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))


        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:

          for i in range(start_ndx, end_ndx):

            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()
            file_path = os.path.join(TEMP_TARGET_PATH, filenames[i]+'.jpg')
            image_data = tf.gfile.FastGFile(file_path, 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            example = image_to_tfexample( image_data, b'jpg', height, width,
                                    list(labels[i]),
                                    bytes( filenames[i] ))
            tfrecord_writer.write(example.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()


def convert_original_iamge(split_name='train'):

  start_time = datetime.datetime.now()

  if split_name == 'train':     # jpeg for hollywoods ; 
      NUM_SHARDS = 10
      temp = np.loadtxt(TRAIN_IMAGE_PATH, dtype='str')
      img_names = [os.path.join(TRAIN_BASIC_PATH, 'JPEGImages', imgname+'.jpeg') for
                         imgname in temp]
      xml_paths = [os.path.join(TRAIN_BASIC_PATH, 'Annotations', imgname+'.xml') for
                         imgname in temp]
  elif split_name == 'test':
      NUM_SHARDS = 1
      temp = np.loadtxt(TEST_IMAGE_PATH, dtype='str')
      img_names = [os.path.join(TRAIN_BASIC_PATH, 'JPEGImages', imgname+'.jpeg') for
                        imgname in temp]
      xml_paths = [os.path.join(TRAIN_BASIC_PATH, 'Annotations', imgname+'.xml') for
                        imgname in temp]
  elif split_name == 'val':
      NUM_SHARDS = 1
      temp = np.loadtxt(VAL_IMAGE_PATH, dtype='str')
      img_names = [os.path.join(TRAIN_BASIC_PATH, 'JPEGImages', imgname+'.jpeg') for
                        imgname in temp]
      xml_paths = [os.path.join(TRAIN_BASIC_PATH, 'Annotations', imgname+'.xml') for
                        imgname in temp]
  else:
      NUM_SHARDS = 1
      temp = np.loadtxt(global_mat_IMAGE_PATH, dtype='str')
      img_names = [os.path.join(TRAIN_BASIC_PATH, 'JPEGImages', imgname+'.jpeg') for
                        imgname in temp]
      xml_paths = [os.path.join(TRAIN_BASIC_PATH, 'Annotations', imgname+'.xml') for
                        imgname in temp]

  labels = []
  for i in range(len(temp)):
      img_resized, label = prepare_global_label( img_names[i], xml_paths[i])
      # plt.savefig(os.path.join(TEMP_TARGET_PATH,temp[i]+'.jpg'))
      img_resized.save(os.path.join(TEMP_TARGET_PATH,temp[i]+'.jpg') )
      np.savetxt(os.path.join(TEMP_TARGET_PATH,temp[i]+'.txt') ,label)
      # label = np.loadtxt(os.path.join(TEMP_TARGET_PATH,temp[i]+'.txt') )
      labels.append([int(i) for i in label])

  _convert_dataset(split_name, temp, labels, TFRECORD_TARGET_PATH, NUM_SHARDS) # 51.24 122.81

  print('\nFinished converting the dataset to tfrecord!     Time cost ',datetime.datetime.now()-start_time)




if __name__ == '__main__':

    # hollywood data    test:1297, train: 216694 , val: 6676   
    # 这里是源数据文件位置，需要为 VOC 格式
    TRAIN_BASIC_PATH = '/nishome/zl/faster-rcnn/data/HollywoodHeads/'

    TEST_IMAGE_PATH= os.path.join(TRAIN_BASIC_PATH, 'ImageSets/Main/test.txt',)
    TRAIN_IMAGE_PATH = os.path.join(TRAIN_BASIC_PATH, 'ImageSets/Main/train.txt')
    VAL_IMAGE_PATH = os.path.join(TRAIN_BASIC_PATH, 'ImageSets/Main/val.txt')

    # 保存文件位置
    TFRECORD_TARGET_PATH = os.path.join(TRAIN_BASIC_PATH, 'tfrecord')
    TEMP_TARGET_PATH = os.path.join(TRAIN_BASIC_PATH, 'temp') 


    # convert_original_iamge(split_name='test')
    # convert_original_iamge(split_name='val')
    convert_original_iamge(split_name='train')




