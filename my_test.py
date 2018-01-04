# -*- coding:utf-8 -*-

from __future__ import  division
import os
import numpy as np
import tensorflow as tf
import time
import math
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

from global_utils import preprocess_for_global, plot_heatmap, plot_hist
from nets import nets_factory

slim = tf.contrib.slim




def test(CKPT_PATH, plot=True, BATCH_SIZE=1,
         file_pattern=FILE_PATTERN_TEST,
         log_writer=None):

    if file_pattern == FILE_PATTERN_TRAIN:
        print ('Train dataset:')
        num_sample = NUM_SAMPLES_TRAIN
    elif file_pattern == FILE_PATTERN_VAL:
        print ('Val dataset:')
        num_sample = NUM_SAMPLES_VAL
    else:
        print ('Test dataset:')
        num_sample = NUM_SAMPLES

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
                  data_sources=file_pattern,
                  reader=tf.TFRecordReader,
                  num_samples=num_sample,
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
        loss = tf.reduce_mean(
            tf.log(tf.clip_by_value(1 + tf.exp(((-1) ** (rpn_label + 2)) * rpn_cls_score[:, 1]), 1e-12, 1e+12)) +
            tf.log(tf.clip_by_value(1 + tf.exp(((-1) ** (rpn_label + 1)) * rpn_cls_score[:, 0]), 1e-12, 1e+12)))


        rpn_prob = tf.nn.softmax(rpn_cls_score)
        rpn_cls_pred = tf.argmax(rpn_prob, axis=1, name="rpn_cls_pred")

        Accuracy = slim.metrics.streaming_accuracy(rpn_cls_pred, rpn_label)

        saver = tf.train.Saver()

    # Start a new session to show example output.
    with tf.Session(graph=graph) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        last_snapshot = 0
        # restore the ckpt
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            last_snapshot = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('restored from ckpt:',ckpt.model_checkpoint_path, 'last_snapshot:',last_snapshot)
        else:
            print(' CKPT not found ! ')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        num_batches = int(math.ceil(num_sample*NUM_EPOCHS / float(BATCH_SIZE)) )

        loss_list = []
        accu_list = []

        TP = []
        TN = []
        FP = []
        FN = []
        TP_degree = {1:[],2:[],4:[],8:[]}
        for i in range(num_batches):

            np_loss, np_Accuracy, np_filenames, \
            np_rpn_prob, np_rpn_cls_score,\
            np_rpn_cls_pred, np_rpn_label = sess.run([loss, Accuracy,filenames, rpn_prob,
                                                     rpn_cls_score, rpn_cls_pred, rpn_label])

            temp1 = np.where(np_rpn_cls_pred == 1)[0]
            temp2 = np.where(np_rpn_label == 1)[0]
            temp11 = np.where(np_rpn_cls_pred == 0)[0]
            temp21 = np.where(np_rpn_label == 0)[0]
            temp3 = set(temp1).intersection(set(temp2))
            temp4 = set(temp11).intersection(set(temp21))

            TP_per = len(temp3) / len(temp2)
            TN_per = len(temp4) / len(temp21)
            FP_per = (len(temp1)-len(temp3)) / len(temp21)
            FN_per = (len(temp11)-len(temp4)) / len(temp2)

            TP.append(TP_per)
            TN.append(TN_per)
            FP.append(FP_per)
            FN.append(FN_per)

            loss_list.append(np_loss)
            accu_list.append(np_Accuracy[1])

            
            if plot:
                print('process image:',np_filenames[0].decode('utf-8')+'.jpeg', np_rpn_cls_score.shape)
                np_index = [np_Accuracy, TP_per, np_rpn_cls_pred, np_rpn_label, TN_per, FP_per,FN_per]
                plot_heatmap( np_rpn_cls_score[:,1] , np_filenames[0].decode('utf-8')+'.jpeg', np_index, boxes=boxes)

        print('TEST : mean loss:',np.mean(loss_list), ' ,mean Accuracy:',np.mean(accu_list),
              'mean recall_1:', np.mean(TP),'mean recall_0:', np.mean(TN),
              'mean FP:', np.mean(FP), 'mean FN:', np.mean(FN))


        tf.summary.scalar('loss', np.mean(loss_list))
        tf.summary.scalar('Accuracy', np.mean(accu_list))
        tf.summary.scalar('TP', np.mean(TP))
        tf.summary.scalar('TN', np.mean(TN))
        tf.summary.scalar('FP', np.mean(FP))
        tf.summary.scalar('FN', np.mean(FN))
        summary_op = tf.summary.merge_all()
        summary = sess.run(summary_op)
        log_writer.add_summary(summary, last_snapshot)

        plot_hist(TP, name='TP.jpg')
        plot_hist(TN, name='TN.jpg')
        plot_hist(FP, name='FP.jpg')
        plot_hist(FN, name='FN.jpg')

        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':

    TRAIN_BASIC_PATH = '/nishome/zl/faster-rcnn/data/HollywoodHeads/tfrecord'
    IMG_PATH = '/nishome/zl/faster-rcnn/data/HollywoodHeads/JPEGImages'
    XML_PATH = '/nishome/zl/faster-rcnn/data/HollywoodHeads/Annotations'

    TEST_RESULT_PATH = os.path.join(TRAIN_BASIC_PATH, 'test_result')
    if not os.path.exists(TEST_RESULT_PATH):
        os.makedirs(TEST_RESULT_PATH)

    FILE_PATTERN_TRAIN = os.path.join(TRAIN_BASIC_PATH,'train_*.tfrecord')
    FILE_PATTERN_TEST = os.path.join(TRAIN_BASIC_PATH, 'test_*.tfrecord')
    FILE_PATTERN_VAL = os.path.join(TRAIN_BASIC_PATH,'val_*.tfrecord')


    CKPT_PATH = os.path.join(TRAIN_BASIC_PATH, 'last3_add_all_0.001')


    NUM_CLASSES = 284 * 2
    NUM_EPOCHS = 1
    NUM_SAMPLES = 1297
    NUM_SAMPLES_TRAIN = 216694
    NUM_SAMPLES_VAL = 6676
    IS_TRAINING = False
    eval_image_size = 224

    log_writer = tf.summary.FileWriter(os.path.join(TRAIN_BASIC_PATH, 'global_summary', 'test'))

    # while True:
    #     test(CKPT_PATH,plot=False, log_writer=log_writer)
    #     time.sleep(40)

    test(CKPT_PATH, plot=True, log_writer=log_writer)
    # test(TRAIN_BASIC_PATH, plot=True, file_pattern=FILE_PATTERN_VAL)










