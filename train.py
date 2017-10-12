import json
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from keras import backend
from keras.optimizers import SGD
from PIL import Image
from tqdm import tqdm
from hyperdash import Experiment
from keras.models import Sequential
from keras.layers import Activation, Dense, Reshape
from keras.layers import Conv2D


def save_model(model, name):
    json_data = model.to_json()
    with open(name, 'w') as f:
        f.write(json_data)


def build_CNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9, 9), input_shape=(48, 48, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(3, kernel_size=(3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    return model


def load_data(dirname):
    x_train = np.array([])
    y_train1 = np.array([])
    y_train2 = np.array([])
    y_train3 = np.array([])
    y_train4 = np.array([])
    counter = 0
    for file in tqdm(os.listdir(dirname)):
        # 拡張子が.jpgでなければ
        if os.path.splitext(file)[1] != '.jpg':
            continue
        image = Image.open(dirname + '/' + file)
        x_train = np.append(x_train, image.resize((48, 48)))
        image = image.resize((96, 96))
        image = np.array(image)
        y_train1 = np.append(y_train1, image[0::2,0::2])
        y_train2 = np.append(y_train2, image[0::2,1::2])
        y_train3 = np.append(y_train3, image[1::2,0::2])
        y_train4 = np.append(y_train4, image[1::2,1::2])
        counter = counter + 1
    x_train = x_train / 255
    y_train1 = y_train1 / 255
    y_train2 = y_train2 / 255
    y_train3 = y_train3 / 255
    y_train4 = y_train4 / 255
    # 正規化する必要あり
    shape = (counter, 48, 48, 3)
    x_train = x_train.reshape(shape)
    y_train1 = y_train1.reshape(shape)
    y_train2 = y_train2.reshape(shape)
    y_train3 = y_train3.reshape(shape)
    y_train4 = y_train4.reshape(shape)
    return (x_train, y_train1, y_train2, y_train3, y_train4)


def main():
    exp = Experiment('4PCNN')
    (x_train, y_train1, y_train2, y_train3, y_train4) = load_data('images')
    upper_left = build_CNN()
    upper_right = build_CNN()
    lower_left = build_CNN()
    lower_right = build_CNN()
    optimizer = SGD(lr=0.5, momentum=0.9, nesterov=True)
    upper_left.compile(loss='binary_crossentropy', optimizer=optimizer)
    upper_right.compile(loss='binary_crossentropy', optimizer=optimizer)
    lower_left.compile(loss='binary_crossentropy', optimizer=optimizer)
    lower_right.compile(loss='binary_crossentropy', optimizer=optimizer)
    save_model(upper_left, 'ul_model.json')
    save_model(upper_right, 'ur_model.json')
    save_model(lower_left, 'll_model.json')
    save_model(lower_right, 'lr_model.json')
    setting = tf.ConfigProto()
    setting.gpu_options.allow_growth = True
    sess = tf.Session(config=setting)
    backend.set_session(sess)
    upper_left.fit(x_train, y_train1, batch_size=32, epochs=20000)
    upper_right.fit(x_train, y_train2, batch_size=32, epochs=20000)
    lower_left.fit(x_train, y_train3, batch_size=32, epochs=20000)
    lower_right.fit(x_train, y_train4, batch_size=32, epochs=20000)
    upper_left.save_weights('ul_model_weights.hdf5')
    upper_right.save_weights('ur_model_weights.hdf5')
    lower_left.save_weights('ll_model_weights.hdf5')
    lower_right.save_weights('lr_model_weights.hdf5')
    (x_test, y_test1, y_test2, y_test3, y_test4) = load_data('images_test')
    eva = upper_left.evaluate(x_test, y_test1, batch_size=32, verbose=1)
    print(eva)
    eva = upper_right.evaluate(x_test, y_test2, batch_size=32, verbose=1)
    print(eva)
    eva = lower_left.evaluate(x_test, y_test3, batch_size=32, verbose=1)
    print(eva)
    eva = lower_right.evaluate(x_test, y_test4, batch_size=32, verbose=1)
    print(eva)
    exp.end()


if __name__ == '__main__':
    main()