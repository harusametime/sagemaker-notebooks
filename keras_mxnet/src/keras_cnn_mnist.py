import argparse
import keras
import os
import numpy as np

from keras.models import Sequential,  save_mxnet_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

if __name__ == '__main__':
    
    '''
    Put a script for "loading data downloaded from S3" and
    "receiving arguments passed via API"
    '''
        
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--n-class', type=int, default=10)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()
    
    batch_size = args.batch_size
    epochs = args.epochs
    num_classes = args.n_class
    train_dir = args.train
    
    x_train = np.load(os.path.join(train_dir, 'train.npz'))['image']
    y_train = np.load(os.path.join(train_dir, 'train.npz'))['label']
    x_test = np.load(os.path.join(train_dir, 'test.npz'))['image']
    y_test = np.load(os.path.join(train_dir, 'test.npz'))['label']
    
    '''
    Keras script starts here.
    '''
    # input image dimensions
    img_rows, img_cols = 28, 28

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, name='data'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # Instead of Keras model, MXNet model is used for inference.
    keras.models.save_mxnet_model(model=model, prefix=os.path.join(args.model_dir, 'model'))
    
def model_fn(model_dir):
    """
    While Keras is used for training, the gluon model yielded by
    "save_mxnet_model" is used for inference. 
    This is called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    # This requires MXNet => 1.3.0
    import mxnet as mx
    from mxnet import gluon
    net = gluon.nn.SymbolBlock.imports(os.path.join(model_dir, 'model-symbol.json'),
                                   ['/data_input1'], 
                                   param_file=os.path.join(model_dir, 'model-0000.params'),
                                   ctx=mx.cpu())
    return net
