import json
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from keras import backend
from keras.optimizers import SGD, Adam
from PIL import Image
from tqdm import tqdm
from hyperdash import Experiment
from keras.models import Sequential
from keras.layers import Activation, Dense, Reshape
from keras.layers import Conv2D


def save_model(model, name):
    """
    モデルをJSONに変換し、任意の名前で保存する
    # 引数
        model : Keras model
        name : String, 保存先ファイル名
    """
    json = model.to_json()
    with open(name, 'w') as f:
        f.write(json)


def build_model(input_size):
    input_shape = (input_size[0], input_size[1], 3)
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(3, kernel_size=(3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    return model


def load_images(name, size, ext='.jpg'):
    """
    画像群を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
        ext : String, 拡張子
    # 戻り値
        images : Numpy array, 画像データ
    """
    x_images = []
    y_images = []
    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:
            # 拡張子が違うなら処理しない
            continue
        image = Image.open(name+file)
        if image.mode != "RGB":
            # 3ch 画像でなければ変換する
            image.convert("RGB")
        x_image = image.resize((size[0]//2, size[1]//2))
        x_image = image.resize(size, Image.BICUBIC)
        x_image = np.array(x_image)
        y_image = image.resize(size)
        y_image = np.array(y_image)
        x_images.append(x_image)
        y_images.append(y_image)
    # 256階調のデータを0-1の範囲に正規化する
    x_images = np.array(x_images)
    y_images = np.array(y_images)
    x_images = x_images / 255
    y_images = y_images / 255
    return x_images, y_images


def main():
    x_images, y_images = load_images('images/', (128, 128))
    model = build_model((128, 128))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    save_model(model, 'model.json')
    model.fit(x_images, y_images, batch_size=64, epochs=50)
    model.save_weights('weights.hdf5')
    x_test, y_test = load_images('images_sample/', (128, 128))
    eva = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print(eva)


if __name__ == '__main__':
    main()