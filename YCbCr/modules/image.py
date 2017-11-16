import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def to_dirname(name):
    """
    ディレクトリ名の"/"有無の違いを吸収する
    # 引数
        name : String, ディレクトリ名
    # 戻り値
        name : String, 変更後
    """
    if name[-1:] == '/':
        return name
    else:
        return name + '/'


def load_image(name, size):
    """
    画像を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
    # 戻り値
        image : Numpy array, 画像データ
    """
    image = Image.open(name)
    if image.mode != "YCbCr":
        # YCbCr 画像でなければ変換する
        image.convert("YCbCr")
    image = image.resize((size[0]//2, size[1]//2))
    image = image.resize(size, Image.NEAREST)
    image = np.array(image)
    # 正規化
    image = image / 255
    # YとCbCr成分を分割する。
    image = image.transpose((2, 0, 1))
    y_image = image[0::3].transpose((1, 2, 0))
    c_image = image[1:].transpose((1, 2, 0))
    # モデルの入力次元にあわせる
    y_image = np.array([y_image])
    c_image = np.array([c_image])
    return y_image, c_image


def load_images(name, size, ext='.jpg'):
    """
    画像群を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
        ext : String, 拡張子
    # 戻り値
        x_images : Numpy array, 学習画像データ
        y_images : Numpy array, 正解画像データ
    """
    x_images = []
    y_images = []
    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:
            # 拡張子が違うなら処理しない
            continue
        image = Image.open(name+file)
        if image.mode != "YCbCr":
            # YCbCr 画像でなければ変換する
            image.convert("YCbCr")
        # 縮小してから拡大する
        x_image = image.resize((size[0]//2, size[1]//2))
        x_image = x_image.resize(size, Image.NEAREST)
        x_image = np.array(x_image)
        y_image = image.resize(size)
        y_image = np.array(y_image)
        x_images.append(x_image)
        y_images.append(y_image)
    x_images = np.array(x_images)
    y_images = np.array(y_images)
    # 256階調のデータを0-1の範囲に正規化する
    x_images = x_images / 255
    y_images = y_images / 255
    # Y成分のみ抽出
    x_images = x_images.transpose((3, 0, 1, 2))
    x_images = x_images[1::3].transpose((1, 2, 3, 0))
    y_images = y_images.transpose((3, 0, 1, 2))
    y_images = y_images[1::3].transpose((1, 2, 3, 0))
    return x_images, y_images