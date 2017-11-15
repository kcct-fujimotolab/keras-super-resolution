from argparse import ArgumentParser
from keras.optimizers import SGD, Adam

from networks.conv_3layer import build_model
from modules.file import save_model
from modules.image import load_images, to_dirname
from modules.interface import show


def get_args():
    description = 'Build SRCNN models and train'
    parser = ArgumentParser(description=description)
    parser.add_argument('-z', '--size', type=int, nargs=2, default=[128, 128], help='image size after expansion')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=500, help='number of epochs')
    parser.add_argument('-i', '--input', type=str, default='images', help='data sets path')
    return parser.parse_args()


def main():
    args = get_args()
    # パラメータ設定
    image_size = args.size # 画像サイズ
    batch = args.batch # 勾配更新までの回数
    epochs = args.epoch # データを周回する回数
    input_dirname = to_dirname(args.input) # 読み込み先ディレクトリ
    # トレーニング
    x_images, y_images = load_images(input_dirname, image_size)
    model = build_model(image_size)
    optimizer = Adam(lr=0.01)
    model.compile(loss='mse', optimizer=optimizer)
    save_model(model, 'model.json')
    model.fit(x_images, y_images, batch_size=batch, epochs=epochs)
    model.save_weights('weights.hdf5')
    # テスト
    # x_test, y_test = load_images('images_sample/', (128, 128))
    # eva = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    # print(eva)


if __name__ == '__main__':
    main()