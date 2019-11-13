import numpy as np
import matplotlib.pyplot as plt
import sys

def align_rot_vels(ws_1, ws_2):
    '''
    orthoprocrustes
    :param ws_1:
    :param ws_2:
    :return: R : R @ ws_1 = ws_2
    '''
    m1 = np.mean(ws_1, axis=0)
    m2 = np.mean(ws_2, axis=0)
    n = ws_1.shape[0]
    ws_1_c = ws_1 - np.tile(m1.reshape(1, -1), (n, 1))
    ws_2_c = ws_2 - np.tile(m2.reshape(1, -1), (n, 1))
    u, s, v = np.linalg.svd(ws_1_c.transpose() @ ws_2_c)
    R = u @ v
    return R.transpose(), m2 - R.transpose() @ m1

def read_poses(reloc_path, is_gt):
    f_in = open(reloc_path, 'r')
    poses = []
    for line in f_in:
        line_elems = line.split(' ')
        vals = []
        for l in line_elems:
            if len(l) > 0 and l != '\n':
                # print(len(l))
                # print('|'+l+'|')
                vals.append(float(l))
        if len(vals) > 0:
            poses.append(np.asarray(vals))
        # if len(vals) == 11:
        #     print('err '+line)
    poses = np.asarray(poses)
    if is_gt:
        poses = poses[1::2, :]
    # print(poses.shape)
    return poses

def rotation_angle(R):
    if np.isnan(np.sum(R)):
        return np.NaN
    else:
        return np.arccos(max(min(0.5 * (np.trace(R) - 1.0), 1.0), -1.0))

def check_reloc_acc(kitt_poses_path, reloc_path, is_med=True):
    gt_poses = read_poses(kitt_poses_path, True)
    est_poses = read_poses(reloc_path, False)

    # gt_poses = gt_poses[0:250, :]
    # est_poses = est_poses[0:250, :]

    gt_c = gt_poses.reshape(-1, 3, 4)[:, :, 3]
    est_c = est_poses.reshape(-1, 3, 4)[:, :, 3]
    n = est_c.shape[0]

    n = 250
    est_c = est_c[0:n, :]
    gt_poses = gt_poses[0:n, :]
    est_poses = est_poses[0:n, :]

    R, t = align_rot_vels(est_c, gt_c[0:n, :])
    # print(R)
    # print(t)
    n = est_c.shape[0]
    est_poses = est_poses.reshape(-1, 3, 4)
    for i in range(0, n):
        est_poses[i, :, :] = R @ est_poses[i, :, :]
        tr = est_poses[i, :, 3]
        tr += t
        # print(tr)
        est_poses[i, :, 3] = tr
    est_poses = est_poses.reshape(-1, 12)
    is_valid = np.ones(est_poses.shape[0])
    rot_errs = []
    trans_errs = []
    for i in range(1, len(est_poses)):
        P = est_poses[i].reshape(3, 4)
        P_gt = gt_poses[i].reshape(3, 4)
        if np.linalg.norm(P[0:3, 0:3] - np.eye(3)) < 1e-10:
            is_valid[i] = 0
        else:
            dR = P[0:3, 0:3].T @ P_gt[0:3, 0:3]
            rot_errs.append(np.abs(rotation_angle(dR)))
            t_gt = P_gt[0:3, 3]
            dt = P[0:3, 3] - t_gt
            trans_errs.append(np.linalg.norm(dt) ) #/ np.linalg.norm(t_gt)
    valid_share = np.sum(is_valid)/float(is_valid.shape[0])

    # plt.figure()
    # plt.plot(trans_errs[1:])
    # plt.show()
    if is_med:
        return valid_share, np.median(trans_errs), np.median(rot_errs)
    else:
        return valid_share, np.mean(trans_errs), np.mean(rot_errs)

def check_agg_reloc_acc(mode_id, kitti_poses_path, res_path, is_med=True):
    N = 5
    agg_res = np.zeros((N, 3))
    for trial_id in range(0, N):
        fname = res_path + '/RelocalizationTrajectory' + str(mode_id) + '_' + str(trial_id) + '.txt'
        val_share, med_t, med_r = check_reloc_acc(kitti_poses_path, fname, is_med)
        agg_res[trial_id, 0] = val_share
        agg_res[trial_id, 1] = med_r * 180 / np.pi
        agg_res[trial_id, 2] = med_t

    # print(np.median(agg_res, axis=0))
    print(agg_res)


# check_reloc_acc('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt',
#                 'RelocalizationTrajectory0_0.txt')
def eval_all_pnp(kitti_poses_path, res_path, is_med=True):
    for mode_id in range(0, 4):
        print(mode_id)
        check_agg_reloc_acc(mode_id, kitti_poses_path, res_path, is_med)

if __name__ == '__main__':
    eval_all_pnp('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt',
                 '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/'+sys.argv[1],
                 True)

# val_share, med_t, med_r = check_reloc_acc('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt',
#                                           '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/RelocalizationTrajectory0_1.txt',
#                                           True)
# print(val_share)

# check_agg_reloc_acc(0, '/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt',
#                     '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/reloc_no_po', True)