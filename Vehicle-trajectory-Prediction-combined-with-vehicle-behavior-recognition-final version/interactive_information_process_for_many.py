import numpy as np
import pandas as pd
from glob import glob
import scipy.signal
file_path = np.array(glob("data/real_data/*"))
#print(file_path)

def add_interactive_imformation(file_path):
    for file in file_path:
        cols_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        dataS = pd.read_csv(file, usecols=cols_to_use, converters={'ID': int, 'Local_X': float, 'Local_Y': float,
                                                                       'Length': float, 'Width': float, 'Class': int,
                                                                       'Vel': float, 'Acc': float, 'Angle': float,
                                                                       'Static': int,
                                                                       'Straight': int, 'LeftTurn': int,
                                                                       'RightTurn': int, 'UTurn': int,
                                                                       'LeftChange': int, 'RightChange': int,
                                                                       'Label': int})
        #     for i in range(len(dataS['Label'].values)):
        #         if dataS['Label'].values[i] == ' ':
        #             dataS['Label'].values[i] = '1'
        #     dataS.Label = dataS.Label.values.astype(np.int)
        #     print(dataS)
        Static, Straight, LeftTurn, RightTurn, UTurn, LeftChange, RightChange = 0, 0, 0, 0, 0, 0, 0
        print("data size:", len(dataS['Label'].values))
        for i in range(len(dataS['Label'].values)):
            if dataS['Label'].values[i] == 0:
                Static += 1
                dataS['Label'].values[i] = 7
            if dataS['Label'].values[i] == 1:
                Straight += 1
                dataS['Label'].values[i] = 0
            if dataS['Label'].values[i] == 2:
                LeftTurn += 1
                dataS['Label'].values[i] = 1
            if dataS['Label'].values[i] == 3:
                RightTurn += 1
                dataS['Label'].values[i] = 2
            if dataS['Label'].values[i] == 4:
                UTurn += 1
                dataS['Label'].values[i] = 3
            if dataS['Label'].values[i] == 5:
                LeftChange += 1
                dataS['Label'].values[i] = 4
            if dataS['Label'].values[i] == 6:
                RightChange += 1
                dataS['Label'].values[i] = 5
        #     Label = dataS.Label.unique()
        #     print(Label)
        print(dataS.Label.values)
        print("Static:", Static, "Straight:", Straight, "LeftTurn:", LeftTurn, "RightTurn:", RightTurn, "UTurn:", UTurn,
              "LeftChange:", LeftChange, "RightChange:", RightChange)
        num = 18
        traj = np.zeros((np.shape(np.asarray(dataS))[0], num + (7 * 3 + 6 * 3) * 8))
        traj[:, :num] = np.asarray(dataS).astype(float)
        for k in range(len(dataS)):
            time = dataS[['Global_Time']]
            time = np.asarray(time)[k]
            frame_time = dataS[dataS.Global_Time == time[0]]
            frame_time = np.asarray(frame_time)
            print(time[0], 'have', len(frame_time), 'vehicles')
            if frame_time.size > 1:
                dx = np.zeros(np.shape(frame_time)[0])
                dy = np.zeros(np.shape(frame_time)[0])
                vel = np.zeros(np.shape(frame_time)[0])
                acc = np.zeros(np.shape(frame_time)[0])
                angle = np.zeros(np.shape(frame_time)[0])
                beha = np.zeros(np.shape(frame_time)[0])
                vid = np.zeros(np.shape(frame_time)[0])

                for l in range(np.shape(frame_time)[0]):
                    dx[l] = frame_time[l][2] - traj[k][2]
                    dy[l] = frame_time[l][3] - traj[k][3]
                    vel[l] = frame_time[l][7]
                    acc[l] = frame_time[l][8]
                    angle[l] = frame_time[l][9]
                    beha[l] = frame_time[l][16]
                    vid[l] = frame_time[l][0]
                dist = dx * dx + dy * dy  # 计算某一时刻下目标车辆与周围车辆之间的距离
                dist = np.sqrt(dist)
                lim = 39  # 限定最大的周围车辆数为39，
                if len(dist) > lim:
                    idx = np.argsort(dist)  # 返回一个排序后的数组的索引,从小到大排列
                    dx = np.array([dx[i] for i in idx[:lim]])
                    dy = np.array([dy[i] for i in idx[:lim]])
                    dist = np.array([dist[i] for i in idx[:lim]])
                    vel = np.array([vel[i] for i in idx[:lim]])
                    acc = np.array([acc[i] for i in idx[:lim]])
                    angle = np.array([angle[i] for i in idx[:lim]])
                    beha = np.array([beha[i] for i in idx[:lim]])
                    vid = np.array([vid[i] for i in idx[:lim]])

                # left
                xl = dx[dx < -2]
                yl = dy[dx < -2]
                yl = yl[xl > -6]
                distl = dist[dx < -2]
                distl = distl[xl > -6]
                vell = vel[dx < -2]
                vell = vell[xl > -6]
                accl = acc[dx < -2]
                accl = accl[xl > -6]
                anglel = angle[dx < -2]
                anglel = anglel[xl > -6]
                behal = beha[dx < -2]
                behal = behal[xl > -6]
                vidl = vid[dx < -2]
                vidl = vidl[xl > -6]
                xl = xl[xl > -6]

                # left top
                yl_top = yl[yl > 0]
                xl_top = xl[yl > 0]
                xl_top = xl_top[yl_top < 50]
                distl_top = distl[yl > 0]
                distl_top = distl_top[yl_top < 50]
                vell_top = vell[yl > 0]
                vell_top = vell_top[yl_top < 50]
                accl_top = accl[yl > 0]
                accl_top = accl_top[yl_top < 50]
                anglel_top = anglel[yl > 0]
                anglel_top = anglel_top[yl_top < 50]
                behal_top = behal[yl > 0]
                behal_top = behal_top[yl_top < 50]
                vidl_top = vidl[yl > 0]
                vidl_top = vidl_top[yl_top < 50]
                yl_top = yl_top[yl_top < 50]

                # left bot
                yl_bot = yl[yl < 0]
                xl_bot = xl[yl < 0]
                xl_bot = xl_bot[yl_bot > -50]
                distl_bot = distl[yl < 0]
                distl_bot = distl_bot[yl_bot > -50]
                vell_bot = vell[yl < 0]
                vell_bot = vell_bot[yl_bot > -50]
                accl_bot = accl[yl < 0]
                accl_bot = accl_bot[yl_bot > -50]
                anglel_bot = anglel[yl < 0]
                anglel_bot = anglel_bot[yl_bot > -50]
                behal_bot = behal[yl < 0]
                behal_bot = behal_bot[yl_bot > -50]
                vidl_bot = vidl[yl < 0]
                vidl_bot = vidl_bot[yl_bot > -50]
                yl_bot = yl_bot[yl_bot > -50]

                # center
                xc = dx[dx >= -2]
                yc = dy[dx >= -2]
                distc = dist[dx >= -2]
                velc = vel[dx >= -2]
                accc = acc[dx >= -2]
                anglec = angle[dx >= -2]
                behac = beha[dx >= -2]
                vidc = vid[dx >= -2]

                yc = yc[xc < 2]
                distc = distc[xc < 2]
                velc = velc[xc < 2]
                accc = accc[xc < 2]
                anglec = anglec[xc < 2]
                behac = behac[xc < 2]
                vidc = vidc[xc < 2]
                xc = xc[xc < 2]

                # center top
                yc_top = yc[yc > 0]
                xc_top = xc[yc > 0]
                xc_top = xc_top[yc_top < 50]
                distc_top = distc[yc > 0]
                distc_top = distc_top[yc_top < 50]
                velc_top = velc[yc > 0]
                velc_top = velc_top[yc_top < 50]
                accc_top = accc[yc > 0]
                accc_top = accc_top[yc_top < 50]
                anglec_top = anglec[yc > 0]
                anglec_top = anglec_top[yc_top < 50]
                behac_top = behac[yc > 0]
                behac_top = behac_top[yc_top < 50]
                vidc_top = vidc[yc > 0]
                vidc_top = vidc_top[yc_top < 50]
                yc_top = yc_top[yc_top < 50]

                # center bot
                yc_bot = yc[yc < 0]
                xc_bot = xc[yc < 0]
                xc_bot = xc_bot[yc_bot > -50]
                distc_bot = distc[yc < 0]
                distc_bot = distc_bot[yc_bot > -50]
                velc_bot = velc[yc < 0]
                velc_bot = velc_bot[yc_bot > -50]
                accc_bot = accc[yc < 0]
                accc_bot = accc_bot[yc_bot > -50]
                anglec_bot = anglec[yc < 0]
                anglec_bot = anglec_bot[yc_bot > -50]
                behac_bot = behac[yc < 0]
                behac_bot = behac_bot[yc_bot > -50]
                vidc_bot = vidc[yc < 0]
                vidc_bot = vidc_bot[yc_bot > -50]
                yc_bot = yc_bot[yc_bot > -50]

                # right
                xr = dx[dx < 6]
                yr = dy[dx < 6]
                yr = yr[xr > 2]
                distr = dist[dx < 6]
                distr = distr[xr > 2]
                velr = vel[dx < 6]
                velr = velr[xr > 2]
                accr = acc[dx < 6]
                accr = accr[xr > 2]
                angler = angle[dx < 6]
                angler = angler[xr > 2]
                behar = beha[dx < 6]
                behar = behar[xr > 2]
                vidr = vid[dx < 6]
                vidr = vidr[xr > 2]
                xr = xr[xr > 2]

                # right top
                yr_top = yr[yr > 0]
                xr_top = xr[yr > 0]
                xr_top = xr_top[yr_top < 50]
                distr_top = distr[yr > 0]
                distr_top = distr_top[yr_top < 50]
                velr_top = velr[yr > 0]
                velr_top = velr_top[yr_top < 50]
                accr_top = accr[yr > 0]
                accr_top = accr_top[yr_top < 50]
                angler_top = angler[yr > 0]
                angler_top = angler_top[yr_top < 50]
                behar_top = behar[yr > 0]
                behar_top = behar_top[yr_top < 50]
                vidr_top = vidr[yr > 0]
                vidr_top = vidr_top[yr_top < 50]
                yr_top = yr_top[yr_top < 50]

                # right bot
                yr_bot = yr[yr < 0]
                xr_bot = xr[yr < 0]
                xr_bot = xr_bot[yr_bot > -50]
                distr_bot = distr[yr < 0]
                distr_bot = distr_bot[yr_bot > -50]
                velr_bot = velr[yr < 0]
                velr_bot = velr_bot[yr_bot > -50]
                accr_bot = accr[yr < 0]
                accr_bot = accr_bot[yr_bot > -50]
                angler_bot = angler[yr < 0]
                angler_bot = angler_bot[yr_bot > -50]
                behar_bot = behar[yr < 0]
                behar_bot = behar_bot[yr_bot > -50]
                vidr_bot = vidr[yr < 0]
                vidr_bot = vidr_bot[yr_bot > -50]
                yr_bot = yr_bot[yr_bot > -50]

                # parameters,挑选距离近的几个车辆
                mini_top = 7
                mini_bot = 6

                # left top
                iy = np.argsort(distl_top)
                iy = iy[0:min(mini_top, len(distl_top))]
                ltop = len(iy)
                xl_top = np.array([xl_top[i] for i in iy])
                yl_top = np.array([yl_top[i] for i in iy])
                distl_top = np.array([distl_top[i] for i in iy])
                vell_top = np.array([vell_top[i] for i in iy])
                accl_top = np.array([accl_top[i] for i in iy])
                anglel_top = np.array([anglel_top[i] for i in iy])
                behal_top = np.array([behal_top[i] for i in iy])
                vidl_top = np.array([vidl_top[i] for i in iy])

                # left bottom
                iy = np.argsort(distl_bot)
                # iy = np.array(list(reversed(iy)))  # 返回一个反转的迭代器,找出后面最近的车辆
                iy = iy[0:min(mini_bot, len(distl_bot))]
                lbot = len(iy)
                xl_bot = np.array([xl_bot[i] for i in iy])
                yl_bot = np.array([yl_bot[i] for i in iy])
                distl_bot = np.array([distl_bot[i] for i in iy])
                vell_bot = np.array([vell_bot[i] for i in iy])
                accl_bot = np.array([accl_bot[i] for i in iy])
                anglel_bot = np.array([anglel_bot[i] for i in iy])
                behal_bot = np.array([behal_bot[i] for i in iy])
                vidl_bot = np.array([vidl_bot[i] for i in iy])

                # center top
                iy = np.argsort(distc_top)
                iy = iy[0:min(mini_top, len(distc_top))]
                ctop = len(iy)
                xc_top = np.array([xc_top[i] for i in iy])
                yc_top = np.array([yc_top[i] for i in iy])
                distc_top = np.array([distc_top[i] for i in iy])
                velc_top = np.array([velc_top[i] for i in iy])
                accc_top = np.array([accc_top[i] for i in iy])
                anglec_top = np.array([anglec_top[i] for i in iy])
                behac_top = np.array([behac_top[i] for i in iy])
                vidc_top = np.array([vidc_top[i] for i in iy])

                # center bottom
                iy = np.argsort(distc_bot)
                # iy = np.array(list(reversed(iy)))
                iy = iy[0:min(mini_bot, len(distc_bot))]
                cbot = len(iy)
                xc_bot = np.array([xc_bot[i] for i in iy])
                yc_bot = np.array([yc_bot[i] for i in iy])
                distc_bot = np.array([distc_bot[i] for i in iy])
                velc_bot = np.array([velc_bot[i] for i in iy])
                accc_bot = np.array([accc_bot[i] for i in iy])
                anglec_bot = np.array([anglec_bot[i] for i in iy])
                behac_bot = np.array([behac_bot[i] for i in iy])
                vidc_bot = np.array([vidc_bot[i] for i in iy])

                # right top
                iy = np.argsort(distr_top)
                iy = iy[0:min(mini_top, len(distr_top))]
                rtop = len(iy)
                xr_top = np.array([xr_top[i] for i in iy])
                yr_top = np.array([yr_top[i] for i in iy])
                distr_top = np.array([distr_top[i] for i in iy])
                velr_top = np.array([velr_top[i] for i in iy])
                accr_top = np.array([accr_top[i] for i in iy])
                angler_top = np.array([angler_top[i] for i in iy])
                behar_top = np.array([behar_top[i] for i in iy])
                vidr_top = np.array([vidr_top[i] for i in iy])

                # right bottom
                iy = np.argsort(distr_bot)
                #  iy = np.array(list(reversed(iy)))
                iy = iy[0:min(mini_bot, len(distr_bot))]
                rbot = len(iy)
                xr_bot = np.array([xr_bot[i] for i in iy])
                yr_bot = np.array([yr_bot[i] for i in iy])
                distr_bot = np.array([distr_bot[i] for i in iy])
                velr_bot = np.array([velr_bot[i] for i in iy])
                accr_bot = np.array([accr_bot[i] for i in iy])
                angler_bot = np.array([angler_bot[i] for i in iy])
                behar_bot = np.array([behar_bot[i] for i in iy])
                vidr_bot = np.array([vidr_bot[i] for i in iy])
                # 将目标车辆周围各个方向的Id合并如数组
                # traj[k, 68:68 + 6] = np.concatenate((np.zeros(6 - len(xl_bot)), xl_bot))
                # traj[k, 68+6:68 + 12] = np.concatenate((np.zeros(6 - len(yl_bot)), yl_bot))
                # traj[k, 68+12:68 + 18] = np.concatenate((np.zeros(6 - len(distl_bot)), distl_bot))
                # traj[k, 68+18:68+24] = np.concatenate((np.zeros(6 - len(vidl_bot)), vidl_bot))
                # left bot
                for i in range(lbot):
                    traj[k, num + i * 8] = vidl_bot[i]
                    traj[k, num + 1 + i * 8] = xl_bot[i]
                    traj[k, num + 2 + i * 8] = yl_bot[i]
                    traj[k, num + 3 + i * 8] = distl_bot[i]
                    traj[k, num + 4 + i * 8] = vell_bot[i]
                    traj[k, num + 5 + i * 8] = accl_bot[i]
                    traj[k, num + 6 + i * 8] = anglel_bot[i]
                    traj[k, num + 7 + i * 8] = behal_bot[i]

                if lbot < mini_bot:
                    for i in range(mini_bot - lbot):
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + i * 8] = 0
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + 1 + i * 8] = 0
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + 2 + i * 8] = 0
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + 3 + i * 8] = 0
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + 4 + i * 8] = 0
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + 5 + i * 8] = 0
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + 6 + i * 8] = 0
                        traj[k, num + 7 + (lbot - 1) * 8 + 1 + 7 + i * 8] = 0
                    # left top
                for i in range(ltop):
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + i * 8] = vidl_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 1 + i * 8] = xl_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 2 + i * 8] = yl_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 3 + i * 8] = distl_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 4 + i * 8] = vell_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 5 + i * 8] = accl_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 6 + i * 8] = anglel_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + i * 8] = behal_top[i]
                if ltop < mini_top:
                    for i in range(mini_top - ltop):
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + 2 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + 3 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + 4 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + 5 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + 6 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (ltop - 1) * 8 + 1 + 7 + i * 8] = 0
                # center bot
                for i in range(cbot):
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + i * 8] = vidc_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 1 + i * 8] = xc_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 2 + i * 8] = yc_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 3 + i * 8] = distc_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 4 + i * 8] = velc_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 5 + i * 8] = accc_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 6 + i * 8] = anglec_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + i * 8] = behac_bot[i]
                if cbot < mini_bot:
                    for i in range(mini_bot - cbot):
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + 2 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + 3 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + 4 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + 5 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + 6 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    cbot - 1) * 8 + 1 + 7 + i * 8] = 0
                # center top
                for i in range(ctop):
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + i * 8] = vidc_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 1 + i * 8] = xc_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 2 + i * 8] = yc_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 3 + i * 8] = distc_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 4 + i * 8] = velc_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 5 + i * 8] = accc_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 6 + i * 8] = anglec_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + i * 8] = behac_top[i]
                if ctop < mini_top:
                    for i in range(mini_top - ctop):
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + 2 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + 3 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + 4 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + 5 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + 6 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (ctop - 1) * 8 + 1 + 7 + i * 8] = 0
                # right bot
                for i in range(rbot):
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + i * 8] = vidr_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 1 + i * 8] = xr_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 2 + i * 8] = yr_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 3 + i * 8] = distr_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 4 + i * 8] = velr_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 5 + i * 8] = accr_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 6 + i * 8] = angler_bot[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + i * 8] = behar_bot[i]
                if rbot < mini_bot:
                    for i in range(mini_bot - rbot):
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + 2 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + 3 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + 4 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + 5 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + 6 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         rbot - 1) * 8 + 1 + 7 + i * 8] = 0
                # right top
                for i in range(rtop):
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + i * 8] = vidr_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + 1 + i * 8] = xr_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + 2 + i * 8] = yr_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + 3 + i * 8] = distr_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + 4 + i * 8] = velr_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + 5 + i * 8] = accr_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + 6 + i * 8] = angler_top[i]
                    traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                     mini_bot - 1) * 8 + 1 + 7 + i * 8] = behar_top[i]
                if rtop < mini_top:
                    for i in range(mini_top - rtop):
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + 1 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + 2 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + 3 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + 4 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + 5 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + 6 + i * 8] = 0
                        traj[k, num + 7 + (mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                    mini_bot - 1) * 8 + 1 + 7 + (mini_top - 1) * 8 + 1 + 7 + (
                                         mini_bot - 1) * 8 + 1 + 7 + (rtop - 1) * 8 + 1 + 7 + i * 8] = 0
        columns = ['ID', 'Global_Time', 'Local_X', 'Local_Y', 'Length', 'Width', 'Class', 'Vel', 'Acc', 'Angle',
                   'Static', 'Straight', 'Left_Turn', 'Right_Turn', 'U_Turn', 'Left_Change', 'Right_Change', 'Label']
        # left bot
        left_bot = []
        for i in range(1, mini_bot + 1):
            vidl_bot_ = 'Vidl_bot_{}'.format(i)
            left_bot.append(vidl_bot_)
            xl_bot_ = 'Xl_bot_{}'.format(i)
            left_bot.append(xl_bot_)
            yl_bot_ = 'Yl_bot_{}'.format(i)
            left_bot.append(yl_bot_)
            distl_bot_ = 'Distl_bot_{}'.format(i)
            left_bot.append(distl_bot_)
            vell_bot_ = 'vell_bot_{}'.format(i)
            left_bot.append(vell_bot_)
            accl_bot_ = 'accl_bot_{}'.format(i)
            left_bot.append(accl_bot_)
            anglel_bot_ = 'anglel_bot_{}'.format(i)
            left_bot.append(anglel_bot_)
            behal_bot_ = 'behal_bot_{}'.format(i)
            left_bot.append(behal_bot_)
        # left top
        left_top = []
        for i in range(1, mini_top + 1):
            vidl_top_ = 'Vidl_top_{}'.format(i)
            left_top.append(vidl_top_)
            xl_top_ = 'Xl_top_{}'.format(i)
            left_top.append(xl_top_)
            yl_top_ = 'Yl_top_{}'.format(i)
            left_top.append(yl_top_)
            distl_top_ = 'Distl_top_{}'.format(i)
            left_top.append(distl_top_)
            vell_top_ = 'vell_top_{}'.format(i)
            left_top.append(vell_top_)
            accl_top_ = 'accl_top_{}'.format(i)
            left_top.append(accl_top_)
            anglel_top_ = 'anglel_top_{}'.format(i)
            left_top.append(anglel_top_)
            behal_top_ = 'behal_top_{}'.format(i)
            left_top.append(behal_top_)
        # center bot
        center_bot = []
        for i in range(1, mini_bot + 1):
            vidc_bot_ = 'Vidc_bot_{}'.format(i)
            center_bot.append(vidc_bot_)
            xc_bot_ = 'Xc_bot_{}'.format(i)
            center_bot.append(xc_bot_)
            yc_bot_ = 'Yc_bot_{}'.format(i)
            center_bot.append(yc_bot_)
            distc_bot_ = 'Distc_bot_{}'.format(i)
            center_bot.append(distc_bot_)
            velc_bot_ = 'velc_bot_{}'.format(i)
            center_bot.append(velc_bot_)
            accc_bot_ = 'accc_bot_{}'.format(i)
            center_bot.append(accc_bot_)
            anglec_bot_ = 'anglec_bot_{}'.format(i)
            center_bot.append(anglec_bot_)
            behac_bot_ = 'behac_bot_{}'.format(i)
            center_bot.append(behac_bot_)
        # center top
        center_top = []
        for i in range(1, mini_top + 1):
            vidc_top_ = 'Vidc_top_{}'.format(i)
            center_top.append(vidc_top_)
            xc_top_ = 'Xc_top_{}'.format(i)
            center_top.append(xc_top_)
            yc_top_ = 'Yc_top_{}'.format(i)
            center_top.append(yc_top_)
            distc_top_ = 'Distc_top_{}'.format(i)
            center_top.append(distc_top_)
            velc_top_ = 'velc_top_{}'.format(i)
            center_top.append(velc_top_)
            accc_top_ = 'accc_top_{}'.format(i)
            center_top.append(accc_top_)
            anglec_top_ = 'anglec_top_{}'.format(i)
            center_top.append(anglec_top_)
            behac_top_ = 'behac_top_{}'.format(i)
            center_top.append(behac_top_)
        # right bot
        right_bot = []
        for i in range(1, mini_bot + 1):
            vidr_bot_ = 'Vidr_bot_{}'.format(i)
            right_bot.append(vidr_bot_)
            xr_bot_ = 'Xr_bot_{}'.format(i)
            right_bot.append(xr_bot_)
            yr_bot_ = 'Yr_bot_{}'.format(i)
            right_bot.append(yr_bot_)
            distr_bot_ = 'Distr_bot_{}'.format(i)
            right_bot.append(distr_bot_)
            velr_bot_ = 'velr_bot_{}'.format(i)
            right_bot.append(velr_bot_)
            accr_bot_ = 'accr_bot_{}'.format(i)
            right_bot.append(accr_bot_)
            angler_bot_ = 'angler_bot_{}'.format(i)
            right_bot.append(angler_bot_)
            behar_bot_ = 'behar_bot_{}'.format(i)
            right_bot.append(behar_bot_)
        # right top
        right_top = []
        for i in range(1, mini_top + 1):
            vidr_top_ = 'Vidr_top_{}'.format(i)
            right_top.append(vidr_top_)
            xr_top_ = 'Xr_top_{}'.format(i)
            right_top.append(xr_top_)
            yr_top_ = 'Yr_top_{}'.format(i)
            right_top.append(yr_top_)
            distr_top_ = 'Distr_top_{}'.format(i)
            right_top.append(distr_top_)
            velr_top_ = 'velr_top_{}'.format(i)
            right_top.append(velr_top_)
            accr_top_ = 'accr_top_{}'.format(i)
            right_top.append(accr_top_)
            angler_top_ = 'angler_top_{}'.format(i)
            right_top.append(angler_top_)
            behar_top_ = 'behar_top_{}'.format(i)
            right_top.append(behar_top_)

        columns = columns + left_bot + left_top + center_bot + center_top + right_bot + right_top
        pd_data = pd.DataFrame(traj, columns=columns)
        pd_data.to_csv('data/data_interactive/{}'.format(file.split("/")[2]))

add_interactive_imformation(file_path)