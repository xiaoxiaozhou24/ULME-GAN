"""
Created on Mar 11, 2021
@author: Sirui Sun
"""
import pickle
import random
import numpy as np
import os
from tqdm import tqdm
import glob
import cv2
import re
import pickle
import csv
import pandas as pd
from options import Options
from solvers import create_solver
def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def get_kpl(data_root,nums):
    filepaths = os.listdir(data_root+'/imgs')
    filepaths.sort()
    filepaths_model = os.listdir(data_root+'/imgs2')
    data = dict()
    for i in range(nums):
        if i == nums-1:
            content = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            data[filepaths_model[0][:-4]] = np.array(content)
        else:
            content = [0.00, 0.00, 0.00, 0.00, i/(1.2*nums), 0.00, 0.00, 0.00, i/(1.2*nums), 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            data[filepaths[i][:-4]] = np.array(content)
    save_dict(data, os.path.join(data_root, "aus_openface"))
if __name__ == '__main__':
    opt = Options().parse()
    solver = create_solver(opt)
    get_kpl(opt.data_root,127)
    solver.run_solver()
    if opt.mode == 'test':
        videoWriter_tar_path = opt.results + '\\target.mp4'
        if os.path.exists(videoWriter_tar_path):
            os.remove(videoWriter_tar_path)
        imgs_tar = os.listdir(opt.results)
        imgs_tar.sort(key=lambda x: int(x[-7:-4]))
        for img_tar in imgs_tar:
            frame = cv2.imread(opt.results + '\\' + img_tar)
            frame = cv2.resize(frame,(256,256),cv2.INTER_NEAREST)
            cv2.imwrite(opt.results + '\\' + img_tar,frame)
        if opt.save_video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            fps = 100
            videoWriter_tar = cv2.VideoWriter(videoWriter_tar_path, fourcc, fps, (256, 256))
            for img_tar in imgs_tar:
                frame = cv2.imread(opt.results + '\\' + img_tar)
                videoWriter_tar.write(frame)
            videoWriter_tar.release()
    print('all ok')
