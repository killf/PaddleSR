import PaddleSR
from paddle.fluid.layers import image_resize, conv2d
import paddle.fluid as fluid
import pandas as pd
import numpy as np
import os, cv2


class SRCNN:
    def __init__(self, scale=4):
        self.scale = scale

    def __call__(self, lr):
        x = image_resize(lr, scale=self.scale)
        x = conv2d(x, 64, (9, 9), padding=4, act='relu', name='conv1_1')
        x = conv2d(x, 32, (1, 1), act='relu', name='conv2_1')
        x = conv2d(x, 3, (5, 5), padding=2, name='conv3_1')
        return x

    def train(self):
        pass

    def predict(self):
        pass


class CSVDataSet(PaddleSR.DataSet):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if os.path.exists(os.path.join(data_dir, "train.csv")):
            train_file = os.path.join(data_dir, "train.csv")
            test_file = os.path.join(data_dir, "test.csv")
        else:
            train_file = test_file = os.path.join(data_dir, "dev.csv")

        self.train_ls = pd.read_csv(train_file)
        self.test_ls = pd.read_csv(test_file)

        self.train_length = len(self.train_ls)
        self.test_length = len(self.test_ls)

    def load_image(self, lr_image_path, hr_image_path):
        hr_image_path = os.path.join(self.data_dir, hr_image_path)
        lr_image_path = os.path.join(self.data_dir, lr_image_path)

        scale = 4
        hr_crop_size = 96
        lr_crop_size = hr_crop_size // scale
        lr_img = cv2.imread(lr_image_path)
        hr_img = cv2.imread(hr_image_path)

        lr_w = np.random.randint(lr_img.shape[1] - lr_crop_size + 1, size=1)[0]
        lr_h = np.random.randint(lr_img.shape[0] - lr_crop_size + 1, size=1)[0]

        hr_w = lr_w * scale
        hr_h = lr_h * scale

        lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
        hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

        lr_img_cropped = lr_img_cropped / 255.0
        hr_img_cropped = hr_img_cropped / 255.0

        lr_img_cropped = np.transpose(lr_img_cropped, [2, 0, 1])
        hr_img_cropped = np.transpose(hr_img_cropped, [2, 0, 1])

        return lr_img_cropped, hr_img_cropped

    def train_data(self):
        ids = np.arange(self.train_length)
        np.random.shuffle(ids)

        for i in ids:
            hr_image_path = self.train_ls.loc[i]['hr_image_path']
            lr_image_path = self.train_ls.loc[i]['lr_image_path']

            yield self.load_image(lr_image_path, hr_image_path)

    def test_data(self):
        ids = np.arange(self.test_length)
        np.random.shuffle(ids)

        for i in ids:
            hr_image_path = self.test_ls.loc[i]['hr_image_path']
            lr_image_path = self.test_ls.loc[i]['lr_image_path']

            yield self.load_image(lr_image_path, hr_image_path)


if __name__ == '__main__':
    lr = fluid.layers.data("lr", [3, 96, 96], dtype='float32')
    hr = SRCNN()(lr)

    model = PaddleSR.Model(lr, hr, name='srcnn')

    optimizer = fluid.optimizer.Adam(0.001)
    model.compile(optimizer=optimizer, loss=fluid.layers.square_error_cost)

    folder = "/home/killf/dataset/SingleImageSuperResolution4Scale/SingleImageSuperResolution4Scale"
    dataset = CSVDataSet(folder)

    model.model_path = "/home/killf/dlab/PaddleSR/example/srcnn.model"
    model.fit_dataset(dataset, 10, 32, 1)
