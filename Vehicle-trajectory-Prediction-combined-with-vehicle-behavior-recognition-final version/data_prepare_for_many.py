# -*- coding: utf-8 -*-
from io import open
import os.path
from os import path
import random
import numpy as np
import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from glob import glob
file_path = np.array(glob('data/data_interactive/train/*'))

class TrajectoryDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length=40, predict_length=30, file_path=file_path):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """

        self.X_frames_trajectory = []
        self.Y_frames_trajectory = []
        self.length = length
        self.predict_length = predict_length
        for csv_file in file_path:
            self.csv_file = csv_file
            self.load_data()
        self.normalize_data()

    def __len__(self):
        return len(self.X_frames_trajectory)

    def __getitem__(self, idx):
        single_trajectory_data = self.X_frames_trajectory[idx]
        single_trajectory_label = self.Y_frames_trajectory[idx]
        return (single_trajectory_data, single_trajectory_label)

    def load_data(self):
        dataS = pd.read_csv(self.csv_file)
        dataS = dataS[dataS.Class == 5]
        dataS = dataS[dataS.Static == 0]
        dataS = dataS[dataS.Label != 7]
        dataS = dataS[dataS.ID != -1]
        count_ = []
        for vid in dataS.ID.unique():  # 保证是同车的轨迹,一辆车一辆车地加载数据
            frame_ori = dataS[dataS.ID == vid]  # 访问每一辆车的数据
            frame = frame_ori[
                ['Local_X', 'Local_Y', 'Vel', 'Acc', 'Angle', 'Xl_bot_1', 'Yl_bot_1', 'Distl_bot_1', 'vell_bot_1',
                 'accl_bot_1', 'anglel_bot_1', 'Xl_top_1', 'Yl_top_1', 'Distl_top_1', 'vell_top_1', 'accl_top_1',
                 'anglel_top_1', 'Xc_bot_1', 'Yc_bot_1', 'Distc_bot_1', 'velc_bot_1', 'accc_bot_1', 'anglec_bot_1',
                 'Xc_top_1', 'Yc_top_1', 'Distc_top_1', 'velc_top_1', 'accc_top_1', 'anglec_top_1', 'Xr_bot_1',
                 'Yr_bot_1', 'Distr_bot_1', 'velr_bot_1', 'accr_bot_1', 'angler_bot_1', 'Xr_top_1', 'Yr_top_1',
                 'Distr_top_1', 'velr_top_1', 'accr_top_1', 'angler_top_1','Label']]
            frame = np.asarray(frame)
            if len(frame) < 62:
                continue
            dis = frame[1:, :2] - frame[:-1, :2]
            dis = dis.astype(np.float64)
            dis = np.sqrt(np.power(dis[:, 0], 2) + np.power(dis[:, 1], 2))
            idx = np.where(dis > 10)

            if not (idx[0].all):
                print("discontinuious trajectory")
                continue
            frame[:, 0:2] = scipy.signal.savgol_filter(frame[:, 0:2], window_length=21, polyorder=3, axis=0)
            #plt.plot(frame[:,0],frame[:,1],c='r',label='trajectory')
           # plt.legend()
           # plt.show()
            All_vels = []
            for i in range(1):
                x_vel = (frame[1:, 0 + i * 5] - frame[:-1, 0 + i * 5]) / 0.1;  # 计算x方向的速度 ,x: 0,5,10,15,20,25,30
                v_avg = (x_vel[1:] + x_vel[:-1]) / 2.0;
                v_begin = [2.0 * x_vel[0] - v_avg[0]];
                v_end = [2.0 * x_vel[-1] - v_avg[-1]];
                velx = (v_begin + v_avg.tolist() + v_end)
                velx = np.array(velx)

                y_vel = (frame[1:, 1 + i * 5] - frame[:-1, 1 + i * 5]) / 0.1;  # 计算y方向的速度,y:1,6,11,16,21,26,31
                vy_avg = (y_vel[1:] + y_vel[:-1]) / 2.0;
                vy1 = [2.0 * y_vel[0] - vy_avg[0]];
                vy_end = [2.0 * y_vel[-1] - vy_avg[-1]];
                vely = (vy1 + vy_avg.tolist() + vy_end)
                vely = np.array(vely)

                if isinstance(All_vels, (list)):  # 判断一个对象是否是一个已知的类型，类似 type()。
                    All_vels = np.vstack((velx, vely))
                else:
                    All_vels = np.vstack((All_vels, velx.reshape(1, -1)))
                    All_vels = np.vstack((All_vels, vely.reshape(1, -1)))
            All_vels = np.transpose(All_vels)
            total_frame_data = np.concatenate((All_vels[:, :2], frame), axis=1)  # 拼接数组
            if (total_frame_data.shape[0] < 62):
                continue
            X = total_frame_data[:-self.predict_length, :]  # 预测30个轨迹点
            Y = total_frame_data[self.predict_length:, :4]
#             print(X.shape,Y.shape)

            count = 0
            for i in range(X.shape[0] - self.length):
                # if random.random() > 0.2:   #-------------------------------
                # continue
                #                 if count > 60:  # 限制每辆车的最大轨迹数
                #                      break
                #  print('X[] shape',X[i:i+100,:].shape)

                self.X_frames_trajectory = self.X_frames_trajectory + [
                    X[i:i + self.length, :]]  # 生成轨迹段,每个轨迹为100个点,所有轨迹集合,组合成输入数据
                self.Y_frames_trajectory = self.Y_frames_trajectory + [Y[i:i + self.length, :]]  # 生成对应的label
                count = count + 1
            count_.append(count)
            print('File:',self.csv_file.split("/")[3],'Vhicle ID:', vid, " Total trajectory point:", total_frame_data.shape[0], 'Total Trajectory:', count)
        print('Sum Trajectory:',np.sum(count_),'Average Trajectory:', np.mean(count_))
        # print(np.array(self.X_frames_trajectory).shape,np.array(self.Y_frames_trajectory).shape)

    def normalize_data(self):  # 标准化每辆车的输入数据
        # 输出轨迹预测数据，进行标准化
        A = [list(x) for x in zip(*(self.X_frames_trajectory))]
        A = np.array(A).astype(np.float64)
        # A = torch.tensor(A)
        A = torch.from_numpy(A)
        print(A.shape)
        A = A.view(-1, A.shape[2])
        print('A shape:', A.shape)
        if self.csv_file.split("/")[2] == 'train':
            self.mn = torch.mean(A, dim=0)
            # print(self.mn.shape)
            self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values) / 2.0
            self.range = torch.ones(self.range.shape, dtype=torch.double)
            # print(self.range.shape)
            self.std = torch.std(A, dim=0)
            #   print(self.std.shape)
            std = self.std.numpy()    
            mn = self.mn.numpy() 
            rg = self.range.numpy()    
            np.savetxt("std.txt", std)
            np.savetxt("mean.txt", mn)
            np.savetxt("rg.txt", rg)
        else:
            mn= torch.from_numpy(np.loadtxt('mean.txt'))
            std = torch.from_numpy(np.loadtxt('std.txt'))
            rg = torch.from_numpy(np.loadtxt('rg.txt'))
            self.mn = mn
            self.range = rg
            self.std = std
        self.X_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn) / (self.std * self.range) for item in
            self.X_frames_trajectory]
        self.Y_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn[:4]) / (self.std[:4] * self.range[:4]) for
            item in self.Y_frames_trajectory]


def get_dataloader(BatchSize=64, length=40, predict_length=30,file_path = np.array(glob('data/data_interactive/train/*')),daset = 'train'):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    # load dataset
    if path.exists("pickle/dataset_traj_{}_0114_44_4_{}_{}.pickle".format(daset,predict_length, length)):
        with open('pickle/dataset_traj_{}_0114_44_4_{}_{}.pickle'.format(daset,predict_length, length), 'rb') as data:
            dataset = pickle.load(data)
    else:
        dataset = TrajectoryDataset(length, predict_length,file_path)
        with open('pickle/dataset_traj_{}_0114_44_4_{}_{}.pickle'.format(daset,predict_length, length), 'wb') as output:
            pickle.dump(dataset, output)
    # split dataset into train test and validation 8:1:1
    length_traj = dataset.__len__()
    #num_train_traj = (int)(length_traj * 0.8)
   # num_test_traj = (int)(length_traj * 0.9) - num_train_traj
    #num_validation_traj = (int)(length_traj - num_test_traj - num_train_traj)

   # train_traj, test_traj, validation_traj = torch.utils.data.random_split(dataset, [num_train_traj, num_test_traj,
                                                                                    # num_validation_traj])

    train_loader_traj = DataLoader(dataset, batch_size=BatchSize, shuffle=True)
    #test_loader_traj = DataLoader(test_traj, batch_size=BatchSize, shuffle=True)
   # validation_loader_traj = DataLoader(validation_traj, batch_size=BatchSize, shuffle=True)
    iters = iter(train_loader_traj)
    x_trajectory, y_trajectory = next(iters)
    print("*" * 100)
    if daset == 'train':
        print('训练轨迹轨迹条数：', length_traj)
    if daset == 'valid':
        print('验证轨迹轨迹条数：', length_traj)
    if daset == 'test':
        print('测试轨迹轨迹条数：', length_traj)
    print('---轨迹输入数据结构：', x_trajectory.shape, '---轨迹输出数据结构：', y_trajectory.shape)
    print('---轨迹长度：', length, '---预测轨迹长度：', predict_length)
    return (train_loader_traj, dataset)