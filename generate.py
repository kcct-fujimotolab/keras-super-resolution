import os
import numpy as np
from argparse import ArgumentParser
from keras.models import model_from_json
from PIL import Image


def save_images(images, dir_name):
    if os.path.isdir(dir_name) == False:
        os.mkdir(dir_name)
    images = images.astype(np.uint8)
    for i in range(len(images)):
        Image.fromarray(images[i]).save(dir_name + '/result' + str(i) + '.jpg')


def load_model(name):
    with open(name) as f:
        json_data = f.read()
    model = model_from_json(json_data)
    return model


def load_data(dirname):
    x_test = np.array([])
    counter = 0
    for file in tqdm(os.listdir(dirname)):
        # 拡張子が.jpgでなければ
        if os.path.splitext(file)[1] != '.jpg':
            continue
        image = Image.open(dirname + '/' + file)
        x_test = np.append(x_test, image.resize((48, 48)))
        counter = counter + 1
    x_test = x_test / 255
    # 正規化する必要あり
    shape = (counter, 48, 48, 3)
    x_test = x_test.reshape(shape)
    return x_test


def main():
    x_test = load_data('images_test')

    upper_left = load_model('ul_model.json')
    upper_left.load_weights('ul_weights.hdf5')
    upper_right = load_model('ur_model.json')
    upper_right.load_weights('ur_weights.hdf5')
    lower_left = load_model('ll_model.json')
    lower_left.load_weights('ll_weights.hdf5')
    lower_right = load_model('lr_model.json')
    lower_right.load_weights('lr_weights.hdf5')

    gen_images1 = upper_left.predict(x_test, verbose=1)
    gen_images2 = upper_right.predict(x_test, verbose=1)
    gen_images3 = lower_left.predict(x_test, verbose=1)
    gen_images4 = lower_right.predict(x_test, verbose=1)
    gen_images1 = gen_images1 * 255
    gen_images2 = gen_images2 * 255
    gen_images3 = gen_images3 * 255
    gen_images4 = gen_images4 * 255

    images = np.concatenate(gen_images1, gen_images2, gen_images3, gen_images4)
    out_map = images.reshape(len(gen_images1), 2, 2, 48, 48, 3)
    out_map =  out_map.transpose(0, 3, 1, 4, 2, 5)
    out_map = out_map.reshape(len(gen_images1), 96, 96, 3)
    save_images(out_map, 'gen')


if __name__ == '__main__':
    main()