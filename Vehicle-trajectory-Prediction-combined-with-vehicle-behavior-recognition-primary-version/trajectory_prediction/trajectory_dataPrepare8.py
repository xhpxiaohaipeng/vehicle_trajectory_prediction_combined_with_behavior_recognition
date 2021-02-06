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
import numpy as np


class TrajectoryDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length=40, predict_length=30, csv_file='./data/WholeVdata2_interactive_formation_improve.csv'):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """
        self.csv_file = csv_file
        # store X as a list, each element is a 100*42(len * attributes num) np array [velx;vely;x;y;acc;angle] * 7
        self.length = length
        self.predict_length = predict_length
        self.X_frames_trajectory = []
        self.Y_frames_trajectory = []
        self.load_data()
        self.normalize_data()

    def __len__(self):
        return len(self.X_frames_trajectory)

    def __getitem__(self,idx):
        single_trajectory_data = self.X_frames_trajectory[idx]
        single_trajectory_label = self.Y_frames_trajectory[idx]
        return (single_trajectory_data, single_trajectory_label)

    def load_data(self):
        dataS = pd.read_csv(self.csv_file)
        # print(np.unique(dataS['Label']))
        for i in range(len(dataS['Angle'].values)):
            dataS["Angle"].values[i] = dataS["Angle"].values[i] * 180 / np.pi
        # data_label = dataS[["Angle","Label"]]
        # data_label.to_csv("data/label.csv")
        F, L, LO, R, RO = 0, 0, 0, 0, 0
        for i in range(len(dataS['Label'].values)):
            if dataS['Label'].values[i] == 'Follow':
                F += 1
                dataS['Label'].values[i] = 0
            if dataS['Label'].values[i] == 'Left':
                L += 1
                dataS['Label'].values[i] = 1
            if dataS['Label'].values[i] == 'LeftOver':
                LO += 1
                dataS['Label'].values[i] = 2
            if dataS['Label'].values[i] == 'Right':
                R += 1
                dataS['Label'].values[i] = 3
            if dataS['Label'].values[i] == 'RightOver':
                RO += 1
                dataS['Label'].values[i] = 4
        # print(np.unique(dataS['Label']))
        print('Follow:', F, "Left:", L, "LeftOver:", LO, 'Right:', R, "RightOver:", RO)
        max_vehiclenum = np.max(dataS.Vehicle_ID.unique())
        count_ = []
        for vid in dataS.Vehicle_ID.unique():  # 保证是同车的轨迹,一辆车一辆车地加载数据
            # if vid  == 80: #轨迹点数数太少
            #   continue
            #  print('{0} and {1}'.format(vid, max_vehiclenum))
            frame_ori = dataS[dataS.Vehicle_ID == vid]  # 访问每一辆车的数据
            frame = frame_ori[['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Angle',
                                'Xl_bot_1', 'Yl_bot_1', 'Distl_bot_1','Xl_top_1', 'Yl_top_1', 'Distl_top_1', 'Xc_bot_1',                         'Yc_bot_1', 'Distc_bot_1',  'Xc_top_1', 'Yc_top_1', 'Distc_top_1',   'Xr_bot_1', 'Yr_bot_1', 'Distr_bot_1',                      'Xr_top_1', 'Yr_top_1', 'Distr_top_1','Label']]
            frame = np.asarray(frame)
            # print(frame.shape)
            frame[np.where(frame > 4000)] = 0  # assign all 5000 to 0
            # remove anomalies, which has a discontinuious local x or local y
            dis = frame[1:, :2] - frame[:-1, :2]
            dis = dis.astype(np.float64)
            dis = np.sqrt(np.power(dis[:, 0], 2) + np.power(dis[:, 1], 2))
            idx = np.where(dis > 10)
            if not (idx[0].all):
                print("discontinuious trajectory")
                continue
            # smooth the data column wise
            # window size = 5, polynomial order = 3
            # print(frame.shape)
            #  print("beforw filter:",frame[:2])
            frame[:, 0:2] = scipy.signal.savgol_filter(frame[:, 0:2], window_length=51, polyorder=3, axis=0)
            # print("after filter:",frame[:2])
            # print(frame.shape)
            # print(frame[:,0])
            # calculate vel_x and vel_y according to local_x and local_y for all vehi
            All_vels = []
            for i in range(1):
                """
                plt.figure(1)
                plt.subplot(2,3,1)
                print(0+i*4)
                plt.plot(frame[1:,(0+i*4)],'ro')
                plt.subplot(2,3,2)
                plt.plot(frame[1:,1+i*4],'ro')
                plt.show()
                """
                # print(frame.shape)
                x_vel = (frame[1:, 0 + i * 5] - frame[:-1, 0 + i * 5]) / 0.1;  # 计算x方向的速度 ,x: 0,5,10,15,20,25,30
                # print(x_vel.shape,x_vel[:2])
                v_avg = (x_vel[1:] + x_vel[:-1]) / 2.0;
                # print(v_avg.shape)
                v_begin = [2.0 * x_vel[0] - v_avg[0]];
                # print(v_begin)
                v_end = [2.0 * x_vel[-1] - v_avg[-1]];
                velx = (v_begin + v_avg.tolist() + v_end)
                velx = np.array(velx)
                # print(velx.shape)

                y_vel = (frame[1:, 1 + i * 5] - frame[:-1, 1 + i * 5]) / 0.1;  # 计算y方向的速度,y:1,6,11,16,21,26,31
                vy_avg = (y_vel[1:] + y_vel[:-1]) / 2.0;
                vy1 = [2.0 * y_vel[0] - vy_avg[0]];
                vy_end = [2.0 * y_vel[-1] - vy_avg[-1]];
                vely = (vy1 + vy_avg.tolist() + vy_end)
                vely = np.array(vely)  # (n,v_x)
                # print(vely.shape)#(n)
                """
                plt.subplot(2,3,4)
                plt.plot(velx,'r')
                plt.subplot(2,3,5)
                plt.plot(vely,'r')
                plt.subplot(2,3,6)
                print(frame[1:50,0+i*4])
                plt.plot(frame[1:,0+i*4],frame[1:,1+i*4],'g')
                plt.show()
                """
                if isinstance(All_vels, (list)):  # 判断一个对象是否是一个已知的类型，类似 type()。
                    All_vels = np.vstack((velx, vely))
                #    print(All_vels.shape)(2,n)
                else:
                    All_vels = np.vstack((All_vels, velx.reshape(1, -1)))
                    All_vels = np.vstack((All_vels, vely.reshape(1, -1)))
                #  print(All_vels.shape)
            All_vels = np.transpose(All_vels)
            # print(All_vels.shape)#(n,14)
            total_frame_data = np.concatenate((All_vels[:, :2], frame), axis=1)  # 拼接数组
            # total_frame_data = np.concatenate(( All_vels[:,:2], frame, All_vels[:,2:]),axis=1)
            # split into several frames each frame have a total length of 100, drop sequence smaller than 130
            if (total_frame_data.shape[0] < 364):
                continue
            X = total_frame_data[:-self.predict_length, :]  # 预测30个轨迹点
            Y = total_frame_data[self.predict_length:, :4]
            # print(X.shape,Y.shape)

            count = 0
            for i in range(X.shape[0] - self.length):
                if random.random() > 0.2:
                    continue
                j = i - 1;
                if count > 60:  # 限制每辆车的最大轨迹数
                    break
                #  print('X[] shape',X[i:i+100,:].shape)
                self.X_frames_trajectory = self.X_frames_trajectory + [
                    X[i:i + self.length, :]]  # 生成轨迹段,每个轨迹为100个点,所有轨迹集合,组合成输入数据
                self.Y_frames_trajectory = self.Y_frames_trajectory + [Y[i:i + self.length, :]]  # 生成对应的label
                count = count + 1
            count_.append(count)
            print('Vhicle ID:', vid, " Total trajectory point:", total_frame_data.shape[0], 'Total Trajectory:', count)
        print('Average Trajectory:', np.mean(count_))
        # print(np.array(self.X_frames_trajectory).shape,np.array(self.Y_frames_trajectory).shape)

    def normalize_data(self):  # 标准化每辆车的输入数据
        # 输出轨迹预测数据，进行标准化
        A = [list(x) for x in zip(*(self.X_frames_trajectory))]
        A = np.array(A).astype(np.float64)
        # A = torch.tensor(A)
        A = torch.from_numpy(A)
        #  print(A.shape)
        A = A.view(-1, A.shape[2])
        print('A:', A.shape)
        self.mn = torch.mean(A, dim=0)
        # print(self.mn.shape)
        self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values) / 2.0
        self.range = torch.ones(self.range.shape, dtype=torch.double)
        # print(self.range.shape)
        self.std = torch.std(A, dim=0)
        print(self.std[-1])
        #   print(self.std.shape)
        # self.X_frames_trajectory = [torch.tensor(item) for item in self.X_frames_trajectory]
        # self.Y_frames_trajectory = [torch.tensor(item) for item in self.Y_frames_trajectory]

        self.X_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn) / (self.std * self.range) for item in
            self.X_frames_trajectory]
        self.Y_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn[:4]) / (self.std[:4] * self.range[:4]) for
            item in self.Y_frames_trajectory]


def get_dataloader(BatchSize=64, length=40, predict_length=30):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    # load dataset
    if path.exists("pickle/my_dataset_traj_26_4_{}_{}.pickle".format(predict_length, length)):
        with open('pickle/my_dataset_traj_26_4_{}_{}.pickle'.format(predict_length, length), 'rb') as data:
            dataset = pickle.load(data)
    else:
        dataset = TrajectoryDataset(length, predict_length)
        with open('pickle/my_dataset_traj_26_4_{}_{}.pickle'.format(predict_length, length), 'wb') as output:
            pickle.dump(dataset, output)
    # split dataset into train test and validation 8:1:1
    # split dataset into train test and validation 7:2:1
    legth_traj = dataset.__len__()
    num_train_traj = (int)(legth_traj * 0.8)
    num_test_traj = (int)(legth_traj * 0.9) - num_train_traj
    num_validation_traj = (int)(legth_traj - num_test_traj - num_train_traj)

    train_traj, test_traj, validation_traj = torch.utils.data.random_split(dataset, [num_train_traj, num_test_traj, num_validation_traj])

    train_loader_traj = DataLoader(train_traj, batch_size=BatchSize, shuffle=True)
    test_loader_traj = DataLoader(test_traj, batch_size=BatchSize, shuffle=True)
    validation_loader_traj = DataLoader(validation_traj, batch_size=BatchSize, shuffle=True)
    iters = iter(train_loader_traj)
    x_trajectory, y_trajectory = next(iters)
    print("*" * 30, '——-——' * 20, '*' * 30)
    print('---轨迹输入数据结构：', x_trajectory.shape, '---轨迹输出数据结构：', y_trajectory.shape)
    print('---轨迹长度：', length, '---预测轨迹长度：', predict_length)
    return (train_loader_traj, test_loader_traj, validation_loader_traj, dataset)
