import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import torchvision.transforms as transforms


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, mode='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        img_root = '/public/home/v-wangqw/program/CVUSA/'
        if mode == 'train':
            file_list = os.path.join(img_root, 'splits/train-19zl.csv')
        elif mode == 'test':
            file_list = os.path.join(img_root, 'splits/val-19zl.csv')

        data_list = []
        with open(file_list, 'r') as f:
            for line in f:
                data = line.split(',')
                # data_list.append([img_root + data[0], img_root + data[1], img_root + data[2][:-1]])
                data_list.append([img_root + data[0], img_root + data[1], img_root +
                                data[0].replace('bing', 'polar').replace('jpg', 'png')])

        self.aer_list = [item[0] for item in data_list]
        self.pano_list = [item[1] for item in data_list]
        self.polar_list = [item[2] for item in data_list]
        
        self.preprocess_aerialviwe = transforms.Compose([
            # 此步骤后，像素值会在[0, 1]范围内
            transforms.ToTensor(),
            # Normalize步骤会将[0, 1]范围的值转换为[-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # 在这里添加其他必要的转换，例如缩放图像等
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        self.preprocess_streetviwe = transforms.Compose([
            # 此步骤后，像素值会在[0, 1]范围内
            transforms.ToTensor(),
            # Normalize步骤会将[0, 1]范围的值转换为[-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # 在这里添加其他必要的转换，例如缩放图像等
            transforms.Resize(
                (128, 512), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        self.preprocess_polar = transforms.Compose([
            # 此步骤后，像素值会在[0, 1]范围内
            transforms.ToTensor(),
            # Normalize步骤会将[0, 1]范围的值转换为[-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # 在这里添加其他必要的转换，例如缩放图像等
            transforms.Resize(
                (128, 512), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __getitem__(self, index):
        aer_image = Image.open(self.aer_list[index]).convert('RGB')
        pano_image = Image.open(self.pano_list[index]).convert('RGB')
        polar_image = Image.open(self.polar_list[index]).convert('RGB')
        # 应用预处理
        aer_image = self.preprocess_aerialviwe(aer_image)
        pano_image = self.preprocess_streetviwe(pano_image)
        polar_image = self.preprocess_streetviwe(polar_image)
        ground_path = self.pano_list[index]

        return {'ground_path': ground_path, "aer_image": aer_image, "pano_image": pano_image, "polar_image": polar_image}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.aer_list)
