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
import sys
import logging
import os
import re
import time
terminated = False

from sagemaker_tensorflow import PipeModeDataset
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.distributed as dist

class MLP (nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(HEIGHT * WIDTH* DEPTH, 256)   
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, NUM_CLASSES)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    

    # Hyperparameters passed via create-training-job
    parser = argparse.ArgumentParser()
    
    
    # Container environment
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    args, _ = parser.parse_known_args()
    
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

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
    num_epochs = 10
    # This yields 40000 (training images)/64 (batch_size) * 10 (epoch) = 6250 batches (steps)
    # Tensorflow dataset raises tf.errors.OutOfRangeError when all the batches are fed as described in training-loop
    ds = ds.repeat(num_epochs) 
    ds = ds.prefetch(10)
    ds = ds.map(parse, num_parallel_calls=10)
    ds = ds.shuffle(buffer_size = 64) #larger than batch_size
    ds = ds.batch(batch_size = 64)
 
    iterator = ds.make_one_shot_iterator()
    itr_initializer = iterator.make_initializer(ds)
    image_batch, label_batch = iterator.get_next()
    
    # Set up PyTorch Neural Network and optimizer
    net = MLP().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        net = torch.nn.DataParallel(net)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    
    # Let's feed data from TensorFlow dataset into PyTorch Neural net
    with tf.Session() as sess:
        sess.run(itr_initializer)
        total_loss = 0
        step = 0
      
        net.train()
        while True:
            try:
                # Draw batch from Tensorflow tensor
                # sess.run() executes Tensorflow graph including get_next().
                # It is NOT needed to call get_next() here.
                x, y = sess.run([image_batch, label_batch])
                x = torch.from_numpy(x)
                y = torch.from_numpy(y).long()
                

                # Feed into PyTorch neural net
                optimizer.zero_grad()
                output = net(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()

                total_loss += loss
                step += 1
                
                if step % 100 == 0:
                    avg_loss = total_loss/100
                    print("Steps: {}, Average Loss: {}".format(step, avg_loss))
                    total_loss = 0
        
            except tf.errors.OutOfRangeError:
                break
