"""
Created on Mar 11, 2021
@author: Sirui Sun
"""
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
def get_data(filepaths):
    data = dict()
    linear_list = []
    sum = 0
    data1 = []
    for i in range(2,22):
        data1.append(0) 
    for filepath in tqdm(filepaths):
        content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
        for i in range(19,37):
            data1[i-17]= data1[i-17] + content[i]
    data1[18] = data1[19]
    for i in range(2,19):
        print(data1[i]/float(len(filepaths)))
        if data1[i]/float(len(filepaths))<0.5:
            data1[i]=0
        else:
            data1[i]=1
    for filepath in tqdm(filepaths):
        sum = 0
        content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
        for i in range(2,19):
            if opt.re_code:
                if i==20:
                    content[i] = content[i]
                    print(i)               
                else:
                    content[i]=content[i]*data1[i]
            sum = sum + float(content[i])
        linear_list.append(sum)
        if content.ndim == 1:
            data[os.path.basename(filepath[:-4])] = content[2:19]
        elif content.ndim >= 2:
            data[os.path.basename(filepath[:-4])] = content[0][2:19]
    if opt.linear_fitting or opt.apex_frame:
        if opt.top_frame==0:
            opt.top_frame = linear_list.index(max(linear_list))
    print('making datasets...')
    train_path = []
    test_path = []
    test_path_all = os.listdir(opt.data_root+'\imgs')
    if opt.mode == 'test':
        test_path = pd.DataFrame(data=test_path_all)
        test_path.to_csv(opt.data_root + '/test_ids.csv',index=0,header=0)
    else:
        train_path = pd.DataFrame(data=test_path_all[0:int(len(test_path_all)*0.8)])
        train_path.to_csv(opt.data_root + '/train_ids.csv', index=0, header=0)
        test_path = pd.DataFrame(data=test_path_all[int(len(test_path_all)*0.8+1):len(test_path_all)-1])
        test_path.to_csv(opt.data_root + '/test_ids.csv', index=0, header=0)
    print('maked datasets!')
        
    return data

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    opt = Options().parse()
    if opt.pre_treatment:
        print('mesuring au...')
        csv_command1 = 'openface\FaceLandmarkImg.exe -fdir "' + str(opt.data_root) +'\imgs"' +' -aus' + ' -out_dir "'+ str(opt.data_root) + '/csvs"'
        os.system(csv_command1)
        print('mesured au!')
        print('making pkl...')
        filepaths = glob.glob(os.path.join(opt.data_root+'/csvs','*.csv'))
        filepaths.sort()
        data = get_data(filepaths)
        if opt.mode == 'test':
            csv_command2 = 'openface\FaceLandmarkImg.exe -fdir "' + str(opt.data_root) +'\imgs2"' +' -aus' + ' -out_dir "'+ str(opt.data_root) + '/csvs2"'
            os.system(csv_command2)
            filepaths2 = glob.glob(os.path.join(opt.data_root+'/csvs2','*.csv'))
            for filepath in tqdm(filepaths2):
                content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
                data[os.path.basename(filepath[:-4])] = content[2:19]
        save_dict(data, os.path.join(opt.data_root, "aus_openface"))
        print('maked pkl !')
    solver = create_solver(opt)
    solver.run_solver()
    if opt.mode == 'test':
        videoWriter_src_path = opt.results + '\\source.mp4'
        if os.path.exists(videoWriter_src_path):
            os.remove(videoWriter_src_path)
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
            imgs_src = os.listdir(opt.data_root+'/imgs')
            imgs_src.sort(key=lambda x: int(x[-7:-4]))
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            fps = 100
            img = cv2.imread(opt.data_root+'/imgs'+'\\'+imgs_src[0])
            y_src,x_src = img.shape[0:2]
            videoWriter_src = cv2.VideoWriter(videoWriter_src_path, fourcc, fps, (x_src, y_src))
            for img_src in imgs_src:
                frame = cv2.imread(opt.data_root+'/imgs' + '\\' + img_src)
                videoWriter_src.write(frame)
            videoWriter_src.release()
            videoWriter_tar = cv2.VideoWriter(videoWriter_tar_path, fourcc, fps, (256, 256))
            for img_tar in imgs_tar:
                frame = cv2.imread(opt.results + '\\' + img_tar)
                videoWriter_tar.write(frame)
            videoWriter_tar.release()
        

    print('all ok')