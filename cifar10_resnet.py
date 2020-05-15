"""Trains a ResNet on the CIFAR10 dataset.

ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import Constant
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
from advgan_utils import read_from_meta, write_to_meta,get_new_key,Shift_Scale


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

class Subtract_Mean(Layer):

    def __init__(self, bias_initializer, **kwargs):
        super(Subtract_Mean, self).__init__(**kwargs)
        self.bias_initializer = initializers.get(bias_initializer)
        self.class_name = 'Subtract_Mean'
        #self.dtype = dtype

    def build(self, input_shape):

        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            initializer=self.bias_initializer,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)
        super(Subtract_Mean, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.subtract(x, self.bias)

    def get_output_shape(self,input_shape):
        return input_shape
    def get_config(self):
        config = {
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(Subtract_Mean, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
        deserialize_keras_object(    
            config['bias_initializer'],
            module_objects=globals(),
            printable_module_name='initializer')
        return config

def resnet_block(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    if conv_first:
        x = Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4),
                   use_bias=False)(inputs)
        x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x
    x = BatchNormalization()(inputs)
    if activation:
        x = Activation('relu')(x)
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4),
               use_bias=False)(x)
    return x
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
def nn_test():
    input_shape = (32,32,3)
    nn = Sequential()
    nn.add(Flatten())
    nn.add(Dense(128,activation='relu'))
    nn.add(Dense(64,activation='relu'))
    nn.add(Dense(32,activation='relu'))
    nn.add(Dense(10,activation='softmax'))
    return nn


def resnet_v1(input_shape, depth, subtract_pixel_mean=True, num_classes=10,train_data_select=''):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    The number of filters doubles when the feature maps size
    is halved.
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    inputs = Input(shape=input_shape)
    x = inputs
    num_filters = 16
    num_sub_blocks = int((depth - 2) / 6)
    
    #subract the pixel mean of the training set in the first layer
    if subtract_pixel_mean:
        x_train_mean = np.load('saved_models/cifar10'+train_data_select+'_input_mean.npy')
        constant_init = Constant(value=totuple(x_train_mean))
        x=Subtract_Mean(bias_initializer=constant_init)(x)

    x = resnet_block(inputs=x)
    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_block(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if is_first_layer_but_not_first_block:
                x = resnet_block(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, subtract_pixel_mean=True,num_classes=10,train_data_select=''):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    Features maps sizes: 16(input), 64(1st sub_block), 128(2nd), 256(3rd)

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    inputs = Input(shape=input_shape)
    x = inputs
    num_filters_in = 16
    num_filters_out = 64
    filter_multiplier = 4
    num_sub_blocks = int((depth - 2) / 9)

    #subract the pixel mean of the training set in the first layer
    if subtract_pixel_mean:
        x_train_mean = np.load('saved_models/cifar10'+train_data_select+'_input_mean.npy')
        constant_init = Constant(value=totuple(x_train_mean))
        x=Subtract_Mean(bias_initializer=constant_init)(x)
    
    # v2 performs Conv2D on input w/o BN-ReLU
    x = Conv2D(num_filters_in,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4),
               use_bias=False)(x)

    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        if i > 0:
            filter_multiplier = 2
        num_filters_out = num_filters_in * filter_multiplier

        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             conv_first=False)
            y = resnet_block(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_block(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if j == 0:
                x = Conv2D(num_filters_out,
                           kernel_size=1,
                           strides=strides,
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
def resnet_v3(input_shape, depth, subtract_pixel_mean=True, scale_input=True, num_classes=10,train_data_select=''):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-SELU
    Last SELU is after the shortcut connection.
    The number of filters doubles when the feature maps size
    is halved.
    The Number of parameters is approx the same as Table 6 of [a]:

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    inputs = Input(shape=input_shape)
    x = inputs
    num_filters = 16
    num_sub_blocks = int((depth - 2) / 6)
    
    #subract the pixel mean of the training set in the first layer
    if subtract_pixel_mean:
        x_train_mean = np.load('saved_models/cifar10'+train_data_select+'_input_mean.npy')
        constant_init = Constant(value=totuple(x_train_mean))
        x=Subtract_Mean(bias_initializer=constant_init)(x)
    if scale_input:
        """x_train_var = np.load('saved_models/cifar10'+train_data_select+'_input_variance.npy')
        x_train_var = x_train_var()
        print('scaling by x_train_var:')
        x =  Shift_Scale(w=n,b=0)(x)"""

    x = resnet_block(inputs=x)
    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_block(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if is_first_layer_but_not_first_block:
                x = resnet_block(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():


    # Training parameters
    batch_size = 32
    epochs = 200
    data_augmentation = True
    num_classes = 10
    # Train Data Select options:
    #   white_box: 'Full'
    #   black_box_A (First Half): 'A' 
    #   black_box_B (Second Half): 'B' 
    train_data_select = 'B'

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1090Ti
    #           |      | %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  |  3   | 92.16     | 91.25     | -----     | NA        | 35
    # ResNet32  |  5   | 92.46     | 92.49     | -----     | NA        | 50
    # ResNet44  |  7   | 92.50     | 92.83     | -----     | NA        | 70
    # ResNet56  |  9   | 92.71     | 93.03     | 92.60     | NA        | 90 (100)
    # ResNet110 |  18  | 92.65     | 93.39     | 93.03     | 93.63     | 165(180)
    # ---------------------------------------------------------------------------
    n = 6

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 2

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    if version == 2:
        depth = n * 9 + 2
    # Model name, depth and version
    model_type = 'ResNet%d_v%d' % (depth, version)

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()    
    n_samp_half = int(x_train.shape[0]/2)
    if train_data_select == 'A':
        x_train = x_train[:n_samp_half]
        y_train = y_train[:n_samp_half]
        threat_model = 'black_box_A'
    elif train_data_select == 'B':
        x_train = x_train[n_samp_half:]
        y_train = y_train[n_samp_half:]
        threat_model = 'black_box_B'
    else:
        threat_model = 'white_box'
        assert train_data_select=='Full'
        train_data_select = ''


    # Input image dimensions.
    # We assume data format "channels_last".
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]

    if K.image_data_format() == 'channels_first':
        img_rows = x_train.shape[2]
        img_cols = x_train.shape[3]
        channels = x_train.shape[1]
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        img_rows = x_train.shape[1]
        img_cols = x_train.shape[2]
        channels = x_train.shape[3]
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train_var = np.var(x_train,axis=(0,1,2))
        print(x_train_var)
        np.save('saved_models/cifar10'+train_data_select+'_input_mean',x_train_mean)


    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    # Prepare model model saving directory.
    if subtract_pixel_mean == True:
        save_dir = os.path.join('saved_models/cifar10_'+model_type)
    else:
        save_dir = os.path.join('saved_models_NoPixelMean')
    
    model_name = 'cifar10_'+model_type+'_model.{epoch:02d}.h5'    

    ##############get or load meta entry############
    meta = read_from_meta(dataset='CIFAR10')
    meta_key = None
    for k,v in meta['model'].items():
        if v['architecture'] == model_type and v["parent_key"] == "None":
            print('Model Exists for Threat Models: {}'.format(v['threat_models']))
            if input('Train on Dataset{}?(y/n)'.format(train_data_select))=='n':
                quit()
            meta_key = k
            if threat_model not in meta['model'][meta_key]['threat_models']:
                meta['model'][meta_key]['threat_models'].append(threat_model)
    if meta_key is None:
        meta_key = get_new_key('model_new',meta)
        meta['model'].update({meta_key:{
            "architecture": model_type,
            "folder_path": save_dir,
            "parent_key": "None",
            "file_name": model_name.replace('.{epoch:02d}','_clean'),
            "adv_training": False,
            "attacker_key": "clean",
            "threat_models": [
                threat_model
            ],
            "train_params": {
                "batch_size": 128,
                "learning_rate": 0.005,
                "nb_epochs": 200
            },
            "reeval": False,
            "training_finished": False
        }})
    ########################################

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name.replace('cifar10','cifar10'+train_data_select))

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth, subtract_pixel_mean=subtract_pixel_mean,train_data_select=train_data_select)
    elif version == 3:
        model = nn_test()
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth, subtract_pixel_mean=subtract_pixel_mean,train_data_select=train_data_select)

    starting_epoch = False
    if starting_epoch:
        load_path = filepath.replace('epoch','0').format(starting_epoch)
        model.load_weights(load_path)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
    print(model_type)


    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        steps_per_epoch = int(np.ceil(x_train.shape[0] / float(batch_size)))
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=steps_per_epoch,
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    write_to_meta(meta,dataset='CIFAR10')

if __name__ == "__main__":
  main()
