import os
# from argparse import ArgumentParser

from modules.file import load_model
from modules.image import load_image
from modules.interface import show, get_input


def main():
    model = load_model('model.json')
    model.load_weights('weights.hdf5')
    size = (model.input_shape[1], model.input_shape[2])
    print('Enter the file name (*.jpg)')
    while True:
        # 標準入力から取得
        value = get_input()
        # ファイルの存在確認
        if os.path.isfile(value) == False:
            print('File not exist')
            continue
        y_image, c_image = load_image(name=value, size=size)
        show(y_image, c_image)
        prediction = model.predict(y_image)
        show(prediction, c_image)


if __name__ == '__main__':
    main()