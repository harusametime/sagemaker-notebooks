from __future__ import print_function

import argparse
import pickle
import os
import logging
import numpy as np
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__=='__main__':
    
    num_gpus = int(os.environ['SM_NUM_GPUS'])

    parser = argparse.ArgumentParser()
    
    # retrieve the hyperparameters we set from the client (with some defaults)
    parser.add_argument('--algorithm', type=str, default='SVM')
    
    # Data, model, and output directories. These are required.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args, _ = parser.parse_known_args()
    
    X_train = np.load(os.path.join(args.train, 'train.npz'))['feature']
    y_train = np.load(os.path.join(args.train, 'train.npz'))['label']

    X_val = np.load(os.path.join(args.test, 'val.npz'))['feature']
    y_val = np.load(os.path.join(args.test, 'val.npz'))['label']
    
    if args.algorithm == "SVM":
        logger.info("Use SVM as a classifier.")
        from sklearn import svm
        model=svm.SVC()
        model.fit(X_train, y_train)
    elif args.algorithm == "RandomForest":
        logger.info("Use RandomForest as a classifier.")
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier()
        model.fit(X_train, y_train)
        
    
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    logger.info("Accuracy: {}".format(accuracy))

    pickle.dump(model, open(os.path.join(args.model_dir, 'model.pkl'), 'wb'))


def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    This function loads models written during training into `model_dir`.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model
    
    For more on `model_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk
    
    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    with  open(os.path.join(model_dir, 'model.pkl'), 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model

def predict_fn(data, model):
    """A default predict_fn for Chainer is here:
    https://github.com/aws/sagemaker-chainer-container/blob/master/src/sagemaker_chainer_container/serving.py
    Here overrides the function for scikit-learn.
    
    Args:
        data: input data for prediction deserialized by input_fn
        model: model loaded in memory by model_fn
    Returns: a prediction
    """

    return  model.predict(data)