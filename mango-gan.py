import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
from cv2 import cv2

from IPython import display

IMG_SIZE = 224
CHANNELS = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
BATCH_SIZE = 128
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
NOISE_DIM = 100

def get_data():
    train_image_dir = './data/train_dev_data_1/C1-P1_Train'
    train_csv_dir = './data/train_dev_data_1/train.csv'

    # read train.csv
    train_label = []
    with open(train_csv_dir, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            train_label.append([line[0], line[1]])
    train_label = np.array(train_label)         # train_label.shape = (5600, 2)

    def _parse_function_train(filename):
        image_string = tf.io.read_file(train_image_dir + '/' + filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        img = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        img = (img - 127.5) / 127.5     # Normalize to [-1, 1]
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    filenames = tf.constant( train_label[:,0] )
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parse_function_train)
    train_generator = dataset.batch(BATCH_SIZE)

    return train_generator


def make_generator_model():
    # generated image shape = (224, 224, 3), 224 = 2^5 × 7
    model = Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(4, 4), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 112, 112, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 224, 224, CHANNELS)

    return model


def make_discriminator_model():
    model = Sequential()

    model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=IMG_SHAPE, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

checkpoint_dir = './gan_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])


@tf.function
def train_step(images):
    img = cv2.imread('gan-input.jpg')
    img.shape   # (285,399,3)
    img = np.reshape(img, (img.shape[0]*img.shape[1]*img.shape[2], ))
    img.shape   # (341145, )
    img = (img - 127.5) / 127.5     # Normalize to [-1, 1]

    # noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    noise = []
    for i in range(128):
        noise.append(np.random.choice(img, 100))
    noise = np.array(noise)
    noise = tf.convert_to_tensor(noise, dtype='float32')

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('./gan_images/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


if __name__ == '__main__':
    train_generator = get_data()
    train(train_generator, NUM_EPOCHS)
