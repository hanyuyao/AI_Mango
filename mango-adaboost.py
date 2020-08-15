import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier

IMG_SIZE = 224
BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_CLASSES = 3
SHUFFLE_BUFFER_SIZE = 1000
LEARNING_RATE = 1e-5

def get_data():
    train_image_dir = './data/train_dev_data_1/C1-P1_Train'
    train_csv_dir = './data/train_dev_data_1/train.csv'
    test_image_dir = './data/train_dev_data_1/C1-P1_Dev'
    test_csv_dir = './data/train_dev_data_1/dev.csv'

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
        # img = (img - 127.5) / 127.5
        # img = tf.image.random_brightness(img, 0.1)
        # img = tf.image.random_saturation(img, 0.7, 1.3)
        # img = tf.image.random_contrast(img, 0.6, 1.5)
        # img = tf.image.random_flip_left_right(img)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.reshape(img, (IMG_SIZE*IMG_SIZE*3, ))
        return img, label

    def _parse_function_test(filename, label):
        image_string = tf.io.read_file(test_image_dir + '/' + filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        img = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.reshape(img, (IMG_SIZE*IMG_SIZE*3, ))
        return img, label    

    # training data
    filenames = tf.constant( train_label[:,0] )
    temp = train_label[:,1]
    temp = temp.astype('int')
    labels = tf.constant(temp)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function_train)
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    training_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    trainX, trainY = training_iterator.get_next()

    # testing data
    filenames = tf.constant( test_label[:,0] )
    temp = test_label[:,1]
    temp = temp.astype('int')
    labels = tf.constant( temp )
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function_test)
    dataset = dataset.batch(BATCH_SIZE)
    testing_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    testX, testY = testing_iterator.get_next()

    return trainX, trainY, testX, testY


if __name__ == '__main__':
    trainX, trainY, testX, testY = get_data()

    trainX.shape    # (128, 150528)

    # model = ensemble.AdaBoostClassifier( DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=0.6)
    # model = ensemble.AdaBoostClassifier( n_estimators=200 )
    # model = ensemble.BaggingClassifier( n_estimators=500 )

    for x in range(50, 3000, 50):
        model = ensemble.RandomForestClassifier(n_estimators=x)
        model.fit(trainX, trainY)

        prediction = model.predict(testX)

        accuracy = metrics.accuracy_score(testY, prediction)
        print(x, " Accuracy:", accuracy)
        # print("Score: ", model.score(testX, testY))