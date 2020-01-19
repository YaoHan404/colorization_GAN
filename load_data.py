import torch as t
from torch.utils import data
import os
from PIL import Image
import numpy as np
import csv
import random
# 对图像进行处理
from torchvision import transforms as T


class TrainData(data.Dataset):
    """
    继承torch.utils.data.Dtaset, 重写__getitem__()  __len__()
    """
    def __init__(self, root, train=True, use_processed_data=False):
        """
        把图片的路径、label、其它信息按照三元组的形式存储在外存上，
        然后将其全部读入内存。而不是在__getitem__()中一个个按照文件路径来获取图片、读取图片信息，
        这样可以大大减少__getitem__()的时间，也利于数据集的处理。
        :param root: the folder of pictures
        :param train: this dataset is for training or validation
        """
        # 遍历目录下所有文件夹的图片，将所有图片的路径保存在imgs中
        self.img_list_csv = 'train_data.csv'
        # self.root = root
        self.root = 'indoorCVPR_09/Images/'
        if not use_processed_data:
            self.write_csv(root)

        # 读出csv文件
        imgs_list = []
        with open(self.img_list_csv, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                imgs_list.append(row)

        # 定义数据
        if train:
            self.imgs = imgs_list[:int(0.8 * len(imgs_list))]

        else:
            self.imgs = imgs_list[int(0.8 * len(imgs_list)):]

    # __getitem__将多进程并行调用，负载高的函数放在其中可以实现加速
    def __getitem__(self, item):

        # 获取图像路径、速度、label
        img_info = self.imgs[item]
        img_dir = img_info
        pil_img = Image.open(img_dir[0])
        pil_img_rgb = self.transform_data_rgb(pil_img)
        pil_img_gray = self.transform_data_gray(pil_img_rgb)

        return pil_img_gray, pil_img_rgb

    def __len__(self):
        return len(self.imgs)

    def write_csv(self, root):
        folders = os.listdir(root)
        with open(self.img_list_csv, mode='r+') as file:
            file.seek(0)  # 移动文件读取指针到指定位置
            file.truncate()  # 截断文件，即删除其中的内容

            writer = csv.writer(file)

            for folder in folders:
                folder_dir = root+folder+'/'
                one_file_imgs = os.listdir(folder_dir)
                for img in one_file_imgs:  # 读取图片
                    if os.path.splitext(img)[1] == '.jpg':
                        img_dict = folder_dir+img
                        # print(img_dict)
                        writer.writerow([img_dict])

    def transform_data_gray(self, img_input):
        transform = T.Compose([
            # T.Resize(112),  # 长宽比例不变，将最短边变成224 pixel
            # T.CenterCrop(112),  # 从图片中间切出224*224，还有RandomCrop、RandomSizeCrop
            T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2], ), # 将rgb的表示范围从(-1,1)=>(0,1)
            T.ToPILImage(),  # 将tensor转为PILImage
            T.Grayscale(),  # 转灰度图
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5],)  # 标准化至[-1,1], 规定均值和标准差
        ])
        img_output = transform(img_input)
        return img_output

    def transform_data_rgb(self, img_input):
        if img_input.mode != 'RGB':
            img_input = img_input.convert('RGB')
        transform = T.Compose([
            # T.RandomRotation((-10,10), expand=True),  # 随机在-10~10度之间旋转
            T.Resize(130),  # 长宽比例不变，将最短边变成224 pixel
            T.RandomCrop((112,112)),  # 随机剪出 (112,112)
            # T.CenterCrop(112),  # 从图片中间切出224*224，还有RandomCrop、RandomSizeCrop
            T.RandomHorizontalFlip(0.5),  # 随机水平翻转
            T.ToTensor(),  #
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5],)  # 标准化至[-1,1], 规定均值和标准差
        ])
        img_output = transform(img_input)
        return img_output


# '''
# for test
# '''
# train_data_set = TrainData('indoorCVPR_09/Images/')
# # train_data_set.write_csv('indoorCVPR_09/Images/')
# print(train_data_set.__getitem__(0))

# pil_img = Image.open('1.jpg')
# print(len(pil_img.split()))
# print(pil_img.mode)