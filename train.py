import json
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from keras import backend
from keras.optimizers import SGD
from PIL import Image
from tqdm import tqdm
# from hyperdash import Experiment

from keras.models import Sequential
from keras.layers import Activation, Dense, Reshape, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU


def build_CNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9, 9), input_shape=(96, 96, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(Conv2D(3, kernel_size=(3, 3)))
    model.add(Activation('sigmoid'))
    return model


def load_data():
    x_train = np.array([])
    y_train1 = np.array([])
    y_train2 = np.array([])
    y_train3 = np.array([])
    y_train4 = np.array([])
    counter = 0
    for file in tqdm(os.listdir('images_sample')):
        # 拡張子が.jpgでなければ
        if os.path.splitext(file)[1] != '.jpg':
            continue
        image = Image.open('images_sample/' + file)
        x_train = np.append(x_train, image.resize((48, 48)))
        image = image.resize((96, 96))
        image = np.array(image)
        for i in range(96):
            for j in range(96):
                if i % 2 == 0 and j % 2 == 0:
                    y_train1 = np.append(y_train1, image[i][j])
                elif i % 2 == 0 and j % 2 == 1:
                    y_train2 = np.append(y_train2, image[i][j])
                elif i % 2 == 1 and j % 2 == 0:
                    y_train3 = np.append(y_train3, image[i][j])
                elif i % 2 == 1 and j % 2 == 1:
                    y_train4 = np.append(y_train4, image[i][j])
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
    (x_train, y_train1, y_train2, y_train3, y_train4) = load_data()
    image = Image.fromarray(np.uint8(x_train[0] * 255))
    image.show()
    image = Image.fromarray(np.uint8(y_train1[0] * 255))
    image.show()
    image = Image.fromarray(np.uint8(y_train2[0] * 255))
    image.show()
    image = Image.fromarray(np.uint8(y_train3[0] * 255))
    image.show()
    image = Image.fromarray(np.uint8(y_train4[0] * 255))
    image.show()
    # model = build_CNN()
    # optimizer = SGD(lr=0.5, momentum=0.9, nesterov=True)
    # model.complie(loss='binary_crossentropy', optimizer=optimizer)
    # setting = tf.ConfigProto()
    # setting.gpu_options.allow_growth = True
    # sess = tf.Session(config=setting)
    # backend.set_session(sess)
    # model.fit(x_train, y_train, batch_size=24, epochs=10000)


if __name__ == '__main__':
    main()