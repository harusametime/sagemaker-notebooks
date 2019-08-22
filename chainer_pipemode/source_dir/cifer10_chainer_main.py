#     Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#    
#         https://aws.amazon.com/apache-2-0/
#    
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import re
import time
terminated = False

from sagemaker_tensorflow import PipeModeDataset
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)
HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers

class MLP(chainer.Chain):

    def __init__(self, h_units=100, n_out=NUM_CLASSES):
        super(MLP, self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, h_units)
            self.l2 = L.Linear(h_units, h_units)
            self.l3 = L.Linear(h_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

if __name__ == '__main__':
    
    # Hyperparameters passed via create-training-job
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()
    

    # Load data sent via pipe mode using PipeModeDataset
    # https://github.com/aws/sagemaker-tensorflow-extensions
    # The return from PipeModeDatase, "ds" in this code, is TensorFlow Dataset
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    def parse(record):
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])
        image = tf.cast(image,  tf.float32)/255.0
        label = tf.cast(parsed['label'], tf.int32)
        return image, label
    
    ds = PipeModeDataset(channel='train', record_format='TFRecord')
    ds = ds.repeat(1) # This dataset yields data for 1 epoch, and is called for n_epoch (implemented as for-loop)
    ds = ds.prefetch(10)
    ds = ds.map(parse, num_parallel_calls=10)
    ds = ds.shuffle(buffer_size = 64) #larger than batch_size
    ds = ds.batch(batch_size = 64)
 
    iterator = ds.make_one_shot_iterator()
    itr_initializer = iterator.make_initializer(ds)
    image_batch, label_batch = iterator.get_next()
    
    # Set up Chainer Neural Network and optimizer   
    net = MLP()
    optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(net)

    # Let's feed data from TensorFlow dataset into Chainer Neural net
    with tf.Session() as sess:
        num_epochs = 20
        for epoch in range(num_epochs):
            sess.run(itr_initializer)
            total_loss = 0
            step = 0
            while True:
                try:
                    # Draw batch from Tensorflow tensor
                    # By "sess.run()", Tensorflow graph including "get_next" is executed.
                    # It is NOT needed to call get_next() here.
                    x, y = sess.run([image_batch, label_batch])
                    
                    # Feed into Chainer neural net
                    pred = net(x)
                    loss = F.softmax_cross_entropy(pred, y)
                    net.cleargrads()
                    loss.backward()
                    optimizer.update()    
                    
                    total_loss += loss
                    step += 1
                  
                except tf.errors.OutOfRangeError:
                    break

            avg_loss = total_loss/step
            
            print("Epoch: {}, Average Loss: {}".format(epoch, avg_loss))
