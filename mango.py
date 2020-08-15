import os
import csv
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

batch_size= 128
epochs = 200
shuffle_buffer_size = 1000
img_size = 224      #resize to (224,224)
num_classes = 3
learning_rate = 0.0005
weight_decay = 0.0005

base_dir = 'D:\Class\Junior2\AI\Final'
data_dir = os.path.join(base_dir, 'data/train_dev_data_1')
train_csv_dir = os.path.join(data_dir, 'train.csv')
train_image_dir = os.path.join(data_dir, 'C1-P1_Train')
test_csv_dir = os.path.join(data_dir, 'dev.csv')
test_image_dir = os.path.join(data_dir, 'C1-P1_Dev')

def get_data():
    # read train.csv
    train_label = []
    with open(train_csv_dir, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            train_label.append([line[0], line[1]])
    train_label = np.array(train_label)         # train_label.shape = (5600, 2)

    # read dev.csv
    test_label = []
    with open(test_csv_dir, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            test_label.append([line[0], line[1]])
    test_label = np.array(test_label)         # test_label.shape = (800, 2)

    # encode label to 0, 1, 2
    le = LabelEncoder()
    train_label[:,1] = le.fit_transform(train_label[:,1])
    test_label[:,1] = le.fit_transform(test_label[:,1])

    # random access images
    # create training dataset, validation dataset, testing dataset
    def _parse_function_train(filename, label):
        image_string = tf.io.read_file(train_image_dir + '/' + filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        img = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        # rescale to [-1, 1]
        img = (img/127.5) - 1
        # images = tf.image.random_brightness(images, 0.1)
        # images = tf.image.random_saturation(images, 0.7, 1.3)
        # images = tf.image.random_contrast(images, 0.6, 1.5)
        # images = tf.image.random_flip_left_right(images)
        img = tf.image.resize(img, (img_size, img_size))
        return img, label

    def _parse_function_test(filename, label):
        image_string = tf.io.read_file(test_image_dir + '/' + filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        img = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        # rescale to [-1, 1]
        img = (img/127.5) - 1
        # images = tf.image.random_brightness(images, 0.1)
        # images = tf.image.random_saturation(images, 0.7, 1.3)
        # images = tf.image.random_contrast(images, 0.6, 1.5)
        # images = tf.image.random_flip_left_right(images)
        img = tf.image.resize(img, (img_size, img_size))
        return img, label

    # training data
    valid_data_size = 800
    filenames = tf.constant( train_label[0:train_label.shape[0] - valid_data_size, 0] )
    temp_train_label = np.array( train_label[0:train_label.shape[0] - valid_data_size, 1] )
    temp_train_label = tf.keras.utils.to_categorical(temp_train_label, 3)
    labels = tf.constant( temp_train_label )
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function_train)
    training_dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    # training_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # trainX, trainY = iterator.get_next()        # trainX.shape = (128, 224, 224, 3), trainY.shape(128, 3)

    # validation data
    filenames = tf.constant( train_label[train_label.shape[0] - valid_data_size:-1, 0] )
    temp_valid_label = np.array( train_label[train_label.shape[0] - valid_data_size:-1, 1] )
    temp_valid_label = tf.keras.utils.to_categorical(temp_valid_label, 3)
    labels = tf.constant( temp_valid_label )
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function_train)
    validation_dataset = dataset.batch(batch_size)
    # for image_batch, label_batch in validation_dataset.take(1):
    #     pass
    # print(image_batch.shape)                  # (128, 224, 224, 3)

    # validation_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # validX, validY = iterator.get_next()        # validX.shape = (128, 224, 224, 3)


    # testing data
    filenames = tf.constant( test_label[:,0] )
    temp_test_label = np.array( test_label[:,1] )
    temp_test_label = tf.keras.utils.to_categorical(temp_test_label, 3)
    labels = tf.constant( temp_test_label )
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function_test)
    testing_dataset = dataset.batch(batch_size)
    # testing_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # testX, testY = iterator.get_next()        # testX.shape = (128, 224, 224, 3), testY.shape = (128, 3)

    return training_dataset, validation_dataset, testing_dataset


def build_model_1():
    # vgg16
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=(224,224,3),kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    optimizer = optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def build_model_2():
    # original data batch.shape = (128, 224, 224, 3)
    # The feature extractor converts each 224x224x3 image into a 7x7x1280 block of features.
    # through base_model => (128, 7, 7, 1280)
    # through global_average_layer => (128, 1280)
    IMG_SHAPE = (img_size, img_size, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
    # Freezing (by setting layer.trainable = False)
    # prevents the weights in a given layer from being updated during training
    # Use the base model as a feature extractor
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = Dense(num_classes, kernel_initializer='normal', activation='softmax')

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    optimizer = optimizers.Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


def build_model_3():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adadelta(lr=2.0, rho=0.95, epsilon=None, decay=0.0001),
            metrics=['acc'])
    model.summary()
    return model


if __name__ == '__main__':
    # set GPU
    # import os
    # if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    training_dataset, validation_dataset, testing_dataset = get_data()
    model = build_model_2()

    model.fit(training_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            verbose=1)

    clean_acc = model.evaluate(testX, testY, verbose=0)
    print( "testing accuracy: {:.3f} %".format(clean_acc[1] * 100) )
