import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224

test_id = []
for i in range(1, 61):
    test_id.append(str(i) + '.jpg')
test_id = np.array(test_id)

def get_data():
    test_image_dir = './data/testing/Program Exam_Mango Grade Classification'

    def _parse_function_test(filename):
        image_string = tf.io.read_file(test_image_dir + '/' + filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        img = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    filenames = tf.constant( test_id )
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parse_function_test)
    dataset = dataset.batch(test_id.shape[0])
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    testX = iterator.get_next()
    return testX


def pred_model():
    # get prediction of one model
    print('predict with one model')
    model = tf.keras.models.load_model('./trained_model/model-7.h5')

    testX  = get_data()
    testX.shape     # (1600, 224, 224, 3)
    predictions = model.predict(testX)
    predictions = np.argmax(predictions, axis=1)
    predictions = predictions.astype('str')

    dic = {
        '0':'A',
        '1':'B',
        '2':'C'
    }

    for i in range(predictions.shape[0]):
        predictions[i] = dic[predictions[i]]

    with open('pred_output.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['image_id', 'label'])
        for i in range(test_id.shape[0]):
            csv_writer.writerow([test_id[i], predictions[i]])


def pred_models():
    # ensemble learning
    print('-----------------------------------------------------------------')
    print('predict with three models.')
    print('-----------------------------------------------------------------')
    dic = {
        '0':'1',
        '1':'2',
        '2':'3'
    }
    testX = get_data()
    
    def get_pred(model_path):
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(testX)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    pred_1 = get_pred('./trained_model/model-1.h5')
    pred_2 = get_pred('./trained_model/model-6.h5')
    pred_3 = get_pred('./trained_model/model-7.h5')
    predictions = []

    # pred_1[0]
    # pred_2[0]
    # pred_3[0]

    for i in range(testX.shape[0]):
        count = [0, 0, 0]
        count[pred_1[i]] += 1
        count[pred_2[i]] += 1
        count[pred_3[i]] += 1
        label = np.argmax(count)
        predictions.append(label)
    predictions = np.array(predictions)
    predictions = predictions.astype('str')

    for i in range(predictions.shape[0]):
        predictions[i] = dic[predictions[i]]

    with open('pred_output.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['ImageID', 'PredictedLabel'])
        for i in range(test_id.shape[0]):
            csv_writer.writerow([i+1, predictions[i]])


if __name__ == '__main__':
    pred_models()
