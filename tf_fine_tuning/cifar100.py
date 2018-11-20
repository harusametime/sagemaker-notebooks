import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import boto3

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'

LEARNING_RATE = 0.001

            
def model_fn(features, labels, mode, params):   
        
    # Download the pretrained model
    bucket_name = params['bucket_name']
    prefix_name = params['prefix_name']
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket_name).download_file(prefix_name, 'resnet.ckpt')
        print("Pretrained model is downloaded.")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    
    
    # Input Layer
    input_layer = tf.reshape(features[INPUT_TENSOR_NAME], [-1, 32, 32, 3])
    
    # Load Pretrained model
    from  tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_50
    last_layer = resnet_v1_50(input_layer, num_classes=None, scope='resnet_v1_50')
    variables_to_restore = tf.contrib.slim.get_variables_to_restore()
    tf.train.init_from_checkpoint("./resnet.ckpt",{v.name.split(':')[0]: v for v in variables_to_restore if not 'biases' in v.name})
    logits =  tf.reshape(tf.layers.dense(inputs=last_layer[0], units=100), [-1, 100])



    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=100), logits=logits)
            
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss}, every_n_iter=10)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks = [logging_hook])

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 32, 32, 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height':  tf.FixedLenFeature([], tf.int64),
            'width':  tf.FixedLenFeature([], tf.int64),
            'depth':  tf.FixedLenFeature([], tf.int64),
            'label':  tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)})
    
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image.set_shape([32, 32, 3])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'train.tfrecords', batch_size=100)


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.tfrecords', batch_size=100)


def _input_fn(training_dir, training_filename, batch_size=100):
    test_file = os.path.join(training_dir, training_filename)
    filename_queue = tf.train.string_input_producer([test_file])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        capacity=1000 + 3 * batch_size)

    return {INPUT_TENSOR_NAME: images}, labels