import numpy as np
from PIL import Image


def show(y_image, c_image):
    """
    画像データを表示
    # 引数
        image : Numpy array, 画像データ
    """
    y_image = y_image[0] * 255
    c_image = c_image[0] * 255
    y_image = y_image.transpose((2, 0, 1))
    c_image = c_image.transpose((2, 0, 1))
    image = np.concatenate((y_image, c_image))
    image = image.transpose((1, 2, 0))
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()


def get_input():
    """
    標準入力から文字列を取得
    # 戻り値
        value : String, 入力値
    """
    value = input('>> ')
    value = value.rstrip()
    if value == 'q':
        exit()
    return value