import os
import numpy as np
from argparse import ArgumentParser
from keras.models import model_from_json
from PIL import Image


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', nargs=1, type=int)
    parser.add_argument('-o', '--output', nargs=1, type=str)
    return parser.parse_args()


def build_GAN(G, D):
    model = Sequential()
    model.add(G)
    # 判別器を判別に使うために学習は止める
    D.trainable = False
    model.add(D)
    return model


def save_images(images, dir_name, file_name):
    if os.path.isdir(dir_name) == False:
        os.mkdir(dir_name)
    if os.path.isdir(dir_name + '/' + file_name) == False:
        os.mkdir(dir_name + '/' + file_name)
    images = images.astype(np.uint8)
    for i in range(len(images)):
        Image.fromarray(images[i]).save(dir_name + '/' + file_name + '/result' + str(i) + '.jpg')


def generate(G, batch):
    input_dim = G.input_shape[1]
    noise = np.random.uniform(-1, 1, (batch, input_dim))
    gen_images = G.predict(noise, verbose=1)
    gen_images = gen_images * 255
    return gen_images


def load_model(name):
    with open(name) as f:
        json_data = f.read()
    model = model_from_json(json_data)
    return model


def main():
    args = get_args()
    if args.batch:
        batch = args.batch[0]
    else:
        batch = 24
    if args.output:
        dir_name = args.output[0]
    else:
        dir_name = 'gen'
    G = load_model('G_model.json')
    G.load_weights('G_weights.hdf5')
    D = load_model('D_model.json')
    D.load_weights('D_weights.hdf5')
    # GAN = build_GAN(G, D)
    images = generate(G, batch)
    save_images(images, dir_name, 'generate')


if __name__ == '__main__':
    main()