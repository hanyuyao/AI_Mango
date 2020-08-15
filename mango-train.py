import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization


IMG_SIZE = 224
BATCH_SIZE = 128
NUM_EPOCHS = 125
NUM_CLASSES = 3
LEARNING_RATE = 0.0005

def lr_schedule(epoch):
    lrate = LEARNING_RATE
    if epoch > 50:
        lrate = 0.0001
    if epoch > 75:
        lrate = 0.00005
    if epoch > 100:
        lrate = 0.00001
    return lrate


def get_data():
    train_dir = './data/train_dev_data_classified/C1-P1_Train'
    validation_dir = './data/train_dev_data_classified/C1-P1_Dev'

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.7, 1.3],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='categorical'
    )
    validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode='categorical'
    )
    return train_generator, validation_generator


def build_model():
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')

    # Fine Tune
    base_model.trainable = True
    print(len(base_model.layers))   # 22
    fine_tune_at = 8
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    global_average_layer = GlobalAveragePooling2D()
    dense_layer = Dense(512)
    batch_norm_layer = BatchNormalization()
    act_layer = Activation('relu')
    dropout_layer = Dropout(0.4)
    prediction_layer = Dense(NUM_CLASSES, kernel_initializer='normal', activation='softmax')

    model = Sequential([
        base_model,
        global_average_layer,
        dense_layer,
        batch_norm_layer,
        act_layer,
        dropout_layer,
        prediction_layer
    ])

    optimizer = optimizers.Adam(lr=LEARNING_RATE,beta_1=0.9,beta_2=0.999)
    # optimizer = optimizers.RMSprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def build_model_1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # optimizer = optimizers.Adadelta(lr=LEARNING_RATE, rho=0.95, epsilon=None, decay=0.0001)
    optimizer = optimizers.Adam(lr=LEARNING_RATE,beta_1=0.9,beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    train_generator, validation_generator = get_data()
    model = build_model_1()

    callback = []
    callback.append(tf.keras.callbacks.ModelCheckpoint('./train-checkpoint/checkpoint.h5', save_best_only=True, save_weights_only=False))
    callback.append(LearningRateScheduler(lr_schedule))

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.__len__(),          # batch size
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.__len__(),    # batch size
        verbose = 1,
        callbacks=callback
    )

    model.save('my_model.h5') 

