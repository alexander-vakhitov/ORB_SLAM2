import numpy as np

def read_camera_pose(debug_folder, method_label, frame_id):
    f = open(debug_folder + '/_' + method_label + '_traj.txt', 'r')
    cnt = 0
    for line in f:
        if cnt == frame_id:
            data_raw = line.split(' ')[0:12]
            data = []
            for d in data_raw:
                data.append(float(d))
            return np.asarray(data).reshape((3,4))

        cnt += 1

def read_gt_pose(kitti_gt, frame_id):
    f = open(kitti_gt, 'r')
    cnt = 0
    T_last = np.eye(4)
    for line in f:
        data_raw = line.split(' ')[0:12]
        data = []
        for d in data_raw:
            data.append(float(d))
        T34 = np.reshape(np.asarray(data), (3,4))
        T = np.eye(4)
        T[0:3,0:4] = T34
        if cnt == frame_id:
            T_inc = np.linalg.inv(T).dot(T_last)
            return T_inc[0:3,0:4]
        else:
            T_last = T

        cnt += 1

def read_sigma_3d(debug_folder, frame_num):
    f = open(debug_folder + '/' + str(frame_num)+'_S.txt', 'r')
    sigmas = []
    for line in f:
        data_raw = line.split(' ')
        s = []
        for d in data_raw:
            if d == '\n':
                continue
            s.append(float(d))

        s = np.asarray(s)
        S = np.reshape(s,(3,3))
        sigmas.append(S)
    return sigmas


def read_XX(debug_folder, frame_num):
    f = open(debug_folder + '/' + str(frame_num) + '_XX.txt', 'r')
    XX = []
    for line in f:
        data_raw = line.split(' ')
        s = []
        for d in data_raw:
            s.append(float(d))

        s = np.asarray(s)
        XX.append(s)
    return XX

def read_xx(debug_folder, frame_num):
    f = open(debug_folder + '/' + str(frame_num) + '_xx.txt', 'r')
    XX = []
    for line in f:
        data_raw = line.split(' ')
        s = []
        for d in data_raw:
            s.append(float(d))

        s = np.asarray(s)
        XX.append(s)
    return XX

def read_inliers(debug_folder, frame_num, m):
    f = open(debug_folder + '/' + str(frame_num) + '_m_'+str(m)+'.txt', 'r')
    inliers = []
    for line in f:
        data_raw = line.split(' ')
        inliers.append(int(data_raw[0]))
    return inliers

def read_sigmas2d(debug_folder, frame_num):
    f = open(debug_folder + '/' + str(frame_num) + '_s.txt', 'r')
    inliers = []
    for line in f:
        data_raw = line.split(' ')
        inliers.append(float(data_raw[0]))
    return inliers