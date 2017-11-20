import os
from argparse import ArgumentParser
from modules.file import load_model
from modules.image import load_image
from modules.interface import show, get_input


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None)
    return parser.parse_args()


def single(model, name, size):
    if os.path.isfile(name) == False:
        print('File not exist')
        return
    image = load_image(name=name, size=size)
    show(image)
    prediction = model.predict(image)
    show(prediction)


def continuous(model, size):
    print('Enter the file name (*.jpg)')
    while True:
        # 標準入力から取得
        value = get_input()
        # ファイルの存在確認
        if os.path.isfile(value) == False:
            print('File not exist')
            continue
        image = load_image(name=value, size=size)
        show(image)
        prediction = model.predict(image)
        show(prediction)


def main():
    args = get_args()
    if args.input:
        filename = args.input
    else:
        filename = False
    model = load_model('model.json')
    model.load_weights('weights.hdf5')
    # モデルから画像サイズを取得
    size = (model.input_shape[1], model.input_shape[2])
    if filename:
        single(model=model, name=filename, size=size)
    else:
        continuous(model=model, size=size)


if __name__ == '__main__':
    main()