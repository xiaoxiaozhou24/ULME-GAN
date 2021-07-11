from .base_dataset import BaseDataset
import os
import glob
import random
import numpy as np


class ULMEDataset(BaseDataset):
    """docstring for ULMEDataset"""
    def __init__(self):
        super(ULMEDataset, self).__init__()
        
    def initialize(self, opt):
        super(ULMEDataset, self).initialize(opt)
    def get_aus_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        return self.aus_dict[img_id] / 5.0   # norm to [0, 1]

    def make_dataset(self):
        # return all image full path in a list
        imgs_path = []
        assert os.path.isfile(self.imgs_name_file), "%s does not exist." % self.imgs_name_file
        with open(self.imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs_path = [os.path.join(self.imgs_dir, line.strip()) for line in lines]
            imgs_path = sorted(imgs_path)
        return imgs_path

    def __getitem__(self, index):
        if self.opt.mode == 'train':
            img_path = self.imgs_path[index]
            src_img = self.get_img_by_path(img_path)
            src_img_tensor = self.img2tensor(src_img)
            src_aus = self.get_aus_by_path(img_path)
            tar_img_path = random.choice(self.imgs_path)
            tar_img = self.get_img_by_path(tar_img_path)
            tar_img_tensor = self.img2tensor(tar_img)
            tar_aus = self.get_aus_by_path(tar_img_path)
            if not self.opt.no_aus_noise:
                tar_aus = tar_aus + np.random.uniform(-0.1, 0.1, tar_aus.shape)
            data_dict = {'src_img': src_img_tensor, 'src_aus': src_aus, 'tar_img': tar_img_tensor, 'tar_aus': tar_aus, \
                         'src_path': img_path, 'tar_path': tar_img_path}
        else:
            imgs2_path = os.listdir(self.opt.data_root+'\imgs2')#set only one image
            src_img = self.get_img_by_path(self.opt.data_root+'\imgs2\\'+imgs2_path[0])
            src_img_tensor = self.img2tensor(src_img)
            src_aus = self.get_aus_by_path(self.opt.data_root+'\imgs2\\'+imgs2_path[0])
            tar_img_path = self.imgs_path[index]
            tar_img = self.get_img_by_path(tar_img_path)
            tar_img_tensor = self.img2tensor(tar_img)
            tar_aus = self.get_aus_by_path(tar_img_path)
            data_dict = {'src_img': src_img_tensor, 'src_aus': src_aus, 'tar_img': tar_img_tensor, 'tar_aus': tar_aus, \
                     'src_path': tar_img_path, 'tar_path': tar_img_path}

        return data_dict
