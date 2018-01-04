# -*- coding:utf-8 -*-

from __future__ import  division
import os
import numpy as np
import tensorflow as tf
import datetime
import math
from tensorflow.python.ops import math_ops
import sys
import re
import matplotlib.pyplot as plt
import time

from nets import nets_factory

slim = tf.contrib.slim


TRAIN_BASIC_PATH = '/nishome/zl/faster-rcnn/data/HollywoodHeads/tfrecord'
FILE_PATTERN_TRAIN = os.path.join(TRAIN_BASIC_PATH,'train_*.tfrecord')
FILE_PATTERN_TEST = os.path.join(TRAIN_BASIC_PATH,'test_*.tfrecord')
FILE_PATTERN_VAL = os.path.join(TRAIN_BASIC_PATH,'val_*.tfrecord')


SAVE_MODEL_PATH = os.path.join(TRAIN_BASIC_PATH, 'last3_add_all_0.001')
SAVE_MODEL_NAME = os.path.join(SAVE_MODEL_PATH, 'global.ckpt')

CKPT_PATH = None

# TRAINABLE_SCOPES = None
TRAINABLE_SCOPES = ['vgg_19/fc8/weights', 'vgg_19/fc8/biases','vgg_19/fc7/weights', 'vgg_19/fc7/biases','vgg_19/fc6/weights', 'vgg_19/fc6/biases','vgg_19/fc7a/weights', 'vgg_19/fc7a/biases']

LEARNING_RATE = 0.001
# LEARNING_RATE = 0.00001
# LEARNING_RATE = 0.000001



PRE_TRAINED_CKPT_FILE = '/nishome/zl/slim_ai_challenger/vgg_19.ckpt'
exclusions = ['vgg_19/fc7a/weights', 'vgg_19/fc7a/biases','vgg_19/fc8/weights', 'vgg_19/fc8/biases']

NUM_CLASSES = 284*2
BATCH_SIZE = 12
NUM_EPOCHS = 100
NUM_SAMPLES = 216694  # 4500
NUM_SAMPLES_TEST = 1297 # 6676
eval_image_size = 224
IS_TRAINING = True


if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)

def _get_variables_to_train(trainable_scopes=None):
  """Returns a list of variables to train.
  Returns:
    A list of variables to train by the optimizer.
  """
  if trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in trainable_scopes]
  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def get_variables_to_restore(exclusions=None):
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def init_fun(ckpt_file_name, variables_to_restore):
    return slim.assign_from_checkpoint_fn(ckpt_file_name,
                                          variables_to_restore,
                                          ignore_missing_vars=False)


def preprocess_for_global(image, output_height, output_width):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  _R_MEAN = 123.68
  _G_MEAN = 116.78
  _B_MEAN = 103.94
  
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [output_height, output_width])
  image = tf.squeeze(image)
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])



graph = tf.Graph()
with graph.as_default():
    with tf.device('/device:CPU:0'):

        key_to_features = {
              'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
              'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
              'image/height': tf.FixedLenFeature([], tf.int64),
              'image/width': tf.FixedLenFeature([], tf.int64),
              'image/class/label': tf.FixedLenFeature([284], tf.int64),
              'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),#tf.VarLenFeature(tf.string),
        }

        item_to_handlers = {
              'image': slim.tfexample_decoder.Image('image/encoded','image/format'),
              'label': slim.tfexample_decoder.Tensor('image/class/label'),
              'height': slim.tfexample_decoder.Tensor('image/height'),
              'width': slim.tfexample_decoder.Tensor('image/width'),
              'filename': slim.tfexample_decoder.Tensor('image/filename'),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(key_to_features,item_to_handlers)

        dataset = slim.dataset.Dataset(
              data_sources=FILE_PATTERN_TRAIN,
              reader=tf.TFRecordReader,
              num_samples=NUM_SAMPLES,
              decoder=decoder,
              items_to_descriptions={},
              num_classes=NUM_CLASSES)  # Dataset 只是定义了tfrecord的格式属性和解码器

        provider = slim.dataset_data_provider.DatasetDataProvider(
              dataset,
              num_readers=4,
              shuffle=True,
              num_epochs=NUM_EPOCHS,
              common_queue_capacity=BATCH_SIZE*20,
              common_queue_min=BATCH_SIZE*10)

        [image, label, filename] = provider.get(['image', 'label', 'filename'])

        image = preprocess_for_global(image, eval_image_size, eval_image_size)

        images, labels, filenames = tf.train.batch(
            [image, label, filename],
            batch_size=BATCH_SIZE,
            num_threads=4,
            capacity=BATCH_SIZE*5,
            allow_smaller_final_batch=True,
        )

    network_fn = nets_factory.get_network_fn(
        'vgg_19',
        num_classes=NUM_CLASSES,
        weight_decay=0.00004,
        is_training=IS_TRAINING)

    logits, end_points = network_fn(images)

    rpn_cls_score = tf.reshape(logits, [-1, 2])
    rpn_label = tf.reshape(labels, [-1])
    
    # loss = tf.reduce_mean(
    #                  tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                  logits=rpn_cls_score, labels=rpn_label))

    rpn_label = tf.to_float(rpn_label)
    loss = tf.reduce_mean(tf.log( tf.clip_by_value( 1 + tf.exp( ((-1)**(rpn_label+2))*rpn_cls_score[:,1]), 1e-12,1e+12)) +
                          tf.log( tf.clip_by_value( 1 + tf.exp( ((-1)**(rpn_label+1))*rpn_cls_score[:,0]), 1e-12,1e+12) ) )


    tf.summary.scalar('loss', loss)

    rpn_prob = tf.nn.softmax(rpn_cls_score)
    rpn_cls_pred = tf.argmax(rpn_prob, axis=1, name="rpn_cls_pred")

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                               global_step,
                               int(NUM_SAMPLES / BATCH_SIZE),
                               0.94,
                               staircase=True,
                               name='exponential_decay_learning_rate')

    optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=0.9,momentum=0.9,epsilon=1)


    # only train the speciffcial variables or layers
    if TRAINABLE_SCOPES is not None:
        with tf.name_scope('my_train_variables'):
            trainable_variables = _get_variables_to_train(TRAINABLE_SCOPES)
            print('>>>> trainable_variables:',trainable_variables)
            gradients = tf.gradients(loss, trainable_variables)
            gradients = list(zip(gradients, trainable_variables))
            train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
    else:
        print('>>>> trainable_variables should be all !')
        train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=5)

    # metrics
    Accuracy = slim.metrics.streaming_accuracy(rpn_cls_pred, rpn_label)

    is_true_negative = math_ops.logical_and(math_ops.equal(rpn_label, 0),
                                            math_ops.equal(rpn_cls_pred, 0))
    is_false_negative = math_ops.logical_and(math_ops.equal(rpn_label, 1),
                                            math_ops.equal(rpn_cls_pred, 0))
    is_true_postive = math_ops.logical_and(math_ops.equal(rpn_label, 1),
                                            math_ops.equal(rpn_cls_pred, 1))
    is_false_postive = math_ops.logical_and(math_ops.equal(rpn_label, 0),
                                            math_ops.equal(rpn_cls_pred, 1))
    all_postive = tf.reduce_sum( tf.to_int32(math_ops.equal(rpn_label, 1)))
    all_negative = tf.reduce_sum(tf.to_int32(math_ops.equal(rpn_label, 0)))

    TP = tf.reduce_sum(tf.to_int32(is_true_postive)  ) / all_postive
    TN = tf.reduce_sum(tf.to_int32(is_true_negative) ) / all_negative
    FP = tf.reduce_sum(tf.to_int32(is_false_postive) ) / all_negative
    FN = tf.reduce_sum(tf.to_int32(is_false_negative)) / all_postive

    tf.summary.scalar('TP', TP)
    tf.summary.scalar('TN', TN)
    tf.summary.scalar('FP', FP)
    tf.summary.scalar('FN', FN)

    tf.summary.scalar('Accuracy', Accuracy[1])
    summary_op = tf.summary.merge_all()

# Start a new session to show example output.
with tf.Session(graph=graph) as sess:
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    snapshot_step = 0

    if CKPT_PATH is not None:
        # restore the ckpt fileholder
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            variables_to_restore = get_variables_to_restore([])  # use all variables
            print(' <<<< variables_to_restore, should be all :',variables_to_restore)
            init = init_fun(ckpt.model_checkpoint_path,variables_to_restore=variables_to_restore)
            snapshot_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            init(sess)
            print('session restored from CKPT: ',ckpt.model_checkpoint_path,'step is :',snapshot_step)
        else:
            print(' CKPT not found ! ')
    # else:  在 CKPT_PATH 是 None 或者里面没有 CKPT 文件的时候，从预训练模型初始化
            # restore the ckpt file
            if PRE_TRAINED_CKPT_FILE is not None:
                variables_to_restore = get_variables_to_restore(exclusions)
                print('>>>> variables_to_restore:',variables_to_restore)
                init = init_fun(PRE_TRAINED_CKPT_FILE,variables_to_restore=variables_to_restore)
                init(sess)
                print('session restored from PRE_TRAINED :',PRE_TRAINED_CKPT_FILE)
    else:
        # restore the ckpt file
        if PRE_TRAINED_CKPT_FILE is not None:
            variables_to_restore = get_variables_to_restore(exclusions)
            print('>>>> variables_to_restore:',variables_to_restore)
            init = init_fun(PRE_TRAINED_CKPT_FILE,variables_to_restore=variables_to_restore)
            init(sess)
            print('session restored from PRE_TRAINED :',PRE_TRAINED_CKPT_FILE)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    log_writer = tf.summary.FileWriter(os.path.join(TRAIN_BASIC_PATH, 'global_summary'), sess.graph)
    log_writer_test = tf.summary.FileWriter(os.path.join(TRAIN_BASIC_PATH, 'global_summary','test'))

    TP2 = []
    TP3 = []
    num_batches = int(math.ceil(NUM_SAMPLES*NUM_EPOCHS / float(BATCH_SIZE)) )
    for i in range(num_batches):
        i += snapshot_step
        np_loss, _, np_Accuracy, summary, np_rpn_cls_pred, \
        np_rpn_label, np_rpn_cls_score, np_TP = sess.run(
                            [loss, train_op, Accuracy,summary_op,
                             rpn_cls_pred, rpn_label, rpn_cls_score, TP])

        if i % 30 == 0:

            temp1 = np.where(np_rpn_cls_pred == 1)[0]
            temp2 = np.where(np_rpn_label == 1)[0]
            temp3 = set(temp1).intersection(set(temp2))
            TP_per = len(temp3) / len(temp2)
            TP2.append(TP_per)
            TP3.append(np_TP)

            print('Iteration:', i, 'batch recall_1:', TP_per, ' ,loss:',np_loss, ' ,Accuracy:',np_Accuracy,
                   ' np_TP:',np_TP)
            log_writer.add_summary(summary, i)

        if i % 1000 == 0:
            # from my_test import test
            # test(SAVE_MODEL_PATH, plot=False, log_writer=log_writer_test)

            print('>>>>  Saved model! mean TP:',np.mean(TP2), np.mean(TP3) )
            # save model
            saver.save(sess, SAVE_MODEL_NAME, global_step=i)

    coord.request_stop()
    coord.join(threads)

