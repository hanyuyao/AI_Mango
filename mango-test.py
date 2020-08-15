import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

IMG_SIZE = 224

def get_data():
    test_image_dir = './data/AIMango_sample/sample_image'
    test_csv_dir = './data/AIMango_sample/label.csv'

    test_label = []
    with open(test_csv_dir, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            test_label.append([line[0], line[1]])
    test_label = np.array(test_label)         # test_label.shape = (93, 2)

    def _parse_function_test(filename, label):
        image_string = tf.io.read_file(test_image_dir + '/' + filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        img = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        # rescale to [0, 1]
        # img = (img/255.0)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, label

    le = LabelEncoder()
    test_label[:,1] = le.fit_transform(test_label[:,1])

    filenames = tf.constant( test_label[:,0] )
    temp_test_label = np.array( test_label[:,1] )
    temp_test_label = tf.keras.utils.to_categorical(temp_test_label, 3)
    labels = tf.constant( temp_test_label )
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function_test)
    dataset = dataset.batch(test_label.shape[0])
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    testX, testY = iterator.get_next()        # testX.shape = (1, 224, 224, 3), testY.shape = (1, 3)
    return testX, testY


if __name__ == '__main__':
    model = tf.keras.models.load_model('./trained_model/model-7.h5')

    testX, testY = get_data()
    clean_acc = model.evaluate(testX, testY, verbose=1)
    print( "testing accuracy: {:.3f} %".format(clean_acc[1] * 100) )


    # predictions = model_1.predict(testX)
    # print(np.argmax(predictions, axis=1))
