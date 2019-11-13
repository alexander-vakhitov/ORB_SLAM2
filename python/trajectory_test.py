import reader
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import eldrawer

from matplotlib import cm
import matplotlib.colors as colors

# import draw_2d_cov_projs as draw_2d

def plot_camera(ax, T, T_model, col=(0, 0, 0)):
    s = 0.3
    lw = 2.0
    c = -np.transpose(T[0:3,0:3]).dot(T[0:3, 3])
    c = T_model[0:3,0:3].dot(c) + T_model[0:3,3]

    x, y, z = eldrawer.define_ellipsoid(c, 0.025*np.eye(3))
    # x, y, z = eldrawer.define_ellipsoid([0,0,0], np.eye(3), 1.0)
    # ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(cnt), linewidth=0.1, alpha=1, shade=True)
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=col, linewidth=0.5, alpha=0.5)

    return


    corners = []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            corner = np.asarray([s*i,s*j,s])
            corner = np.transpose(T[0:3,0:3]).dot(corner-T[0:3, 3])
            corners.append(T_model[0:3,0:3].dot(corner)+T_model[0:3,3])

    for i in range(0, 4):
        crn = corners[i]
        ax.plot([c[0], crn[0]], [c[1], crn[1]], [c[2], crn[2]], color=col, linewidth=lw)

    c0 = corners[0]
    c1 = corners[1]
    c2 = corners[2]
    c3 = corners[3]
    ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]], color=col, linewidth=lw)
    ax.plot([c2[0], c3[0]], [c2[1], c3[1]], [c2[2], c3[2]], color=col, linewidth=lw)
    ax.plot([c3[0], c1[0]], [c3[1], c1[1]], [c3[2], c1[2]], color=col,linewidth=lw)
    ax.plot([c2[0], c0[0]], [c2[1], c0[1]], [c2[2], c0[2]], color=col,linewidth=lw)

def plot_camera_2d(ax, T, T_model, col=(0, 0, 0), lbl = '', T_prev=np.zeros((4,4))):
    s = 4*0.3
    lw = 8.0
    c = -np.transpose(T[0:3,0:3]).dot(T[0:3, 3])
    c = T_model[0:3,0:3].dot(c) + T_model[0:3,3]
    # if np.linalg.norm(T_prev)>0 :
    # //    cp = -np.transpose(T_prev[0:3, 0:3]).dot(T_prev[0:3, 3])
    #     cp = T_model[0:3, 0:3].dot(cp) + T_model[0:3, 3]
    #     ax.plot([cp[0], c[0]], [cp[2], c[2]], color=col, marker='o', linewidth=lw, markersize=25)
    # else:
    #     ax.plot(c[0], c[2], color=col, marker='o', linewidth=lw, markersize=25)
    corners = []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            corner = np.asarray([s*i,s*j,s])
            corner = np.transpose(T[0:3,0:3]).dot(corner-T[0:3, 3])
            corners.append(T_model[0:3,0:3].dot(corner)+T_model[0:3,3])

    for i in range(0, 4):
        crn = corners[i]
        ax.plot([c[0], crn[0]], [c[2], crn[2]], color=col, linewidth=lw)

    c0 = corners[0]
    c1 = corners[1]
    c2 = corners[2]
    c3 = corners[3]
    ax.plot([c0[0], c1[0]], [c0[2], c1[2]], color=col, linewidth=lw)
    ax.plot([c2[0], c3[0]], [c2[2], c3[2]], color=col, linewidth=lw)
    ax.plot([c3[0], c1[0]], [c3[2], c1[2]], color=col,linewidth=lw)
    hndl, = ax.plot([c2[0], c0[0]], [c2[2], c0[2]], color=col,linewidth=lw, label=lbl)
    return hndl


def increment_pose(T_gt_m,T_gt):
    T_gt[0:3, 0:3] = T_gt_m[0:3, 0:3].dot(T_gt[0:3, 0:3])
    T_gt[0:3, 3] = T_gt_m[0:3, 0:3].dot(T_gt[0:3, 3]) + T_gt_m[0:3, 3]
    return T_gt

def visualize_full_trajectory(frame_start, frame_end):
    Ts_gt = reader.read_gt_poses_all('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt')
    Ts_gt = np.asarray(Ts_gt)
    cs_gt_cut = Ts_gt[frame_start + 1:frame_end + 1, 0:3, 3]
    T_gt_start = Ts_gt[frame_start]
    #reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_start)
    T_model = np.zeros((3, 4))
    T_model[0:3, 0:3] = T_gt_start[0:3, 0:3].transpose()
    T_model[0:3, 3] = -T_gt_start[0:3, 0:3].transpose().dot(T_gt_start[0:3, 3])

    for i in range(0, cs_gt_cut.shape[0]):
        print(T_model)
        print(cs_gt_cut[i,:])
        cs_gt_cut[i,:] = T_model[0:3,0:3].dot(cs_gt_cut[i,:]) + T_model[0:3,3]
        print(cs_gt_cut[i,:])
        print('-')

    x_min = np.min(cs_gt_cut[:, 0])
    x_max = np.max(cs_gt_cut[:, 0])
    z_min = np.min(cs_gt_cut[:, 2])
    z_max = np.max(cs_gt_cut[:, 2])

    max_depth = z_max

    for frame in range(frame_start, frame_end+1):
        # colors = draw_2d.draw_2d_projs(frame, 2)
        colors = []
        ax, T_gt_prev = visualize_trajectory(frame_start, frame, max_depth)
        T_gt_prev_ext = np.eye(4)
        T_gt_prev_ext[0:3,0:4] = T_gt_prev
        T_gt_prev = np.linalg.inv(T_gt_prev_ext)
        deb_folder = '/home/alexander/materials/pnp3d/data/debug_pnpu/0/'
        Sigmas = reader.read_sigma_3d(deb_folder, frame)
        XX = reader.read_XX(deb_folder, frame)
        dlsu_inliers = reader.read_inliers(deb_folder, frame, m=3)
        # norm = colors.Normalize(vmin=0, vmax=len(XX))
        # cmap = cm.jet
        # m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for cnt in range(0, len(XX)):
            if dlsu_inliers[cnt] == 0:
                continue
            # if np.linalg.norm(XX[cnt]) > 300:
            #     continue
            U2, s2, rotation2 = np.linalg.svd(Sigmas[cnt])
            if np.mean(np.sqrt(s2)) > max_depth:
                continue
            if XX[cnt][2] < 0:
                continue
            if XX[cnt][2] > max_depth:
                continue
            XXc = T_gt_prev[0:3,0:3].dot(XX[cnt]) + T_gt_prev[0:3,3]
            SigmaC = T_gt_prev[0:3,0:3].dot(Sigmas[cnt]).dot(T_gt_prev[0:3,0:3].transpose())
            x, y, z = eldrawer.define_ellipsoid(XXc, SigmaC)
            # x, y, z = eldrawer.define_ellipsoid([0,0,0], np.eye(3), 1.0)
            # ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(cnt), linewidth=0.1, alpha=1, shade=True)
            ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=colors[cnt], linewidth=0.1, alpha=1)
        if not os.path.exists('images/video_3d_2/' + str(frame_start) + '/'):
            os.makedirs('images/video_3d_2/' + str(frame_start) + '/')
        plt.savefig('images/video_3d_2/' + str(frame_start) + '/' + str(frame) + '.png')
        plt.close()


def visualize_trajectory(frame_start, frame_end, max_depth):
    T_gt_start = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_start)
    T_model = np.zeros((3,4))
    T_model[0:3,0:3] = T_gt_start[0:3,0:3].transpose()
    T_model[0:3,3] = -T_gt_start[0:3,0:3].transpose().dot(T_gt_start[0:3,3])
    # max_depth = 7*2
    T_gt = np.zeros((3, 4))
    T_gt[0:3,0:3] = np.eye(3)
    T_dlsu_x_2 = np.zeros((3,4))
    T_dlsu_x_2[0:3, 0:3] = np.eye(3)
    T_ran = np.zeros((3, 4))
    T_ran[0:3, 0:3] = np.eye(3)
    T_epnpu = np.zeros((3, 4))
    T_epnpu[0:3, 0:3] = np.eye(3)
    T_dlsu = np.zeros((3, 4))
    T_dlsu[0:3, 0:3] = np.eye(3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-max_depth / 2, max_depth / 2)
    # ax.set_ylim3d(-max_depth / 4, max_depth / 4)
    ax.set_ylim3d(-max_depth / 2, max_depth / 2)
    ax.set_zlim3d(0, max_depth)
    ax.grid(False)

    T_gt_prev = np.copy(T_gt)

    for frame in range(frame_start, frame_end+1):

        T_ran_m = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'ransac_fin',
                                        frame)
        T_epnpu_m = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                         'epnpu_fin', frame)
        T_dlsu_m = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                             'dlsu_fin', frame)
        T_dlsu_x_2_m = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'dlsu_x_2_fin',
                                         frame)

        T_gt_m = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt',
                                         frame+1)


        # T_gt[0:3,0:3] = T_gt_m[0:3,0:3].dot(T_gt[0:3,0:3])
        # T_gt[0:3, 3] = T_gt_m[0:3, 0:3].dot(T_gt[0:3, 3]) + T_gt_m[0:3,3]
        T_gt_prev = np.copy(T_gt)
        T_gt = increment_pose(T_gt_m, T_gt)
        T_dlsu_x_2 = increment_pose(T_dlsu_x_2_m, T_dlsu_x_2)
        T_ran = increment_pose(T_ran_m, T_ran)
        T_epnpu = increment_pose(T_epnpu_m, T_epnpu)
        T_dlsu = increment_pose(T_dlsu_m, T_dlsu)

        plot_camera(ax, T_gt, T_model, (0, 0, 0))
        plot_camera(ax, T_epnpu, T_model, (0, 1, 0))
        plot_camera(ax, T_dlsu, T_model, (1, 0, 0))
        plot_camera(ax, T_dlsu_x_2, T_model, (1, 1, 0))
        plot_camera(ax, T_ran, T_model, (0, 0, 1))



    return ax, T_gt_prev
        # plt.show()

def build_traj_from_increments(Ts):
    T = np.zeros((3,4))
    T[0:3,0:3] = np.eye(3)
    c_global_lst = []
    for frame in range(0, len(Ts)):
        T_m = Ts[frame]
        T = increment_pose(T_m, T)
        c_global_lst.append(-T[0:3,0:3].transpose().dot(T[0:3,3]))
    return np.asarray(c_global_lst).transpose()

def align_procrustes(cs, cs_gt):
    csm = np.mean(cs, 1)
    csgtm = np.mean(cs_gt, 1)
    dX = np.zeros((3,3))
    for i in range(0, cs.shape[1]):
        dx = cs[:,i] - csm
        dy = cs_gt[:, i] - csgtm
        dX += dx.reshape((3,1)).dot(dy.reshape((1,3)))

    u,s,v = np.linalg.svd(dX)
    # print(u.dot(np.diag(s)).dot(v.transpose()) - dX)
    R = u.dot(v)
    R = R.transpose()
    # print(R.dot(R.transpose()))
    t = -R.dot(csm)+csgtm
    n = cs.shape[1]
    return R.dot(cs)+np.tile(t.reshape((3,1)), (1, n))

def plot_trajs(frame_start, frame_end):
    Ts_gt = reader.read_gt_poses_all('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt')
    Ts_gt = np.asarray(Ts_gt)
    cs_gt_cut = Ts_gt[frame_start+1:frame_end+1, 0:3, 3]
    x_min = np.min(cs_gt_cut[:, 0])
    x_max = np.max(cs_gt_cut[:, 0])
    z_min = np.min(cs_gt_cut[:, 2])
    z_max = np.max(cs_gt_cut[:, 2])
    cs_gt_cut = cs_gt_cut.transpose()

    z_gap = z_max - z_min
    x_gap = x_max - x_min

    w = np.max([x_gap, z_gap])
    z_mid = 0.5*(z_max+z_min)
    z_max = z_mid+w/2
    z_min = z_mid-w/2
    x_mid = 0.5*(x_max+x_min)
    x_max = x_mid+w/2
    x_min = x_mid-w/2
    fig = plt.figure(figsize=(32, 32))
    ax = fig.add_subplot(111)
    ax.set_xlim(x_min - 2, x_max + 2)
    # ax.set_ylim3d(-max_depth / 4, max_depth / 4)
    ax.set_ylim(z_min - 2, z_max + 2)

    Ts_ran = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                          'ransac_fin')
    cs_ran = build_traj_from_increments(Ts_ran)
    cs_ran = align_procrustes(cs_ran[:,frame_start:frame_end], cs_gt_cut)
    Ts_epnpu = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                            'epnpu_fin')
    cs_epnpu = build_traj_from_increments(Ts_epnpu)
    cs_epnpu = align_procrustes(cs_epnpu[:,frame_start:frame_end], cs_gt_cut)
    Ts_dlsu = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                           'dlsu_fin')
    cs_dlsu = build_traj_from_increments(Ts_dlsu)
    Ts_dlsu_x_2 = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                               'dlsu_x_2_fin')
    cs_dlsu = align_procrustes(cs_dlsu[:, frame_start:frame_end], cs_gt_cut)
    cs_dlsu_x_2 = build_traj_from_increments(Ts_dlsu_x_2)
    cs_dlsu_x_2 = align_procrustes(cs_dlsu_x_2[:, frame_start:frame_end], cs_gt_cut)
    plt.plot(cs_ran[0,:], cs_ran[2,:], color=(0,0,1))
    plt.plot(cs_epnpu[0, :], cs_epnpu[2, :], color=(0, 1, 0))
    plt.plot(cs_dlsu[0, :], cs_dlsu[2, :], color=(1, 0, 0))
    plt.plot(cs_dlsu_x_2[0, :], cs_dlsu_x_2[2, :], color=(1, 1, 0))
    plt.plot(cs_gt_cut[0, :], cs_gt_cut[2, :], color=(0, 0, 0))

    plt.show()




def visualize_trajectory_2d(frame_start, frame_end):
    plt.style.use('dark_background')
    if not os.path.exists('images/video_2d/'+str(frame_start)):
        os.makedirs('images/video_2d/'+str(frame_start))
    Ts_gt = reader.read_gt_poses_all('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt')
    Ts_gt = np.asarray(Ts_gt)
    Ts_gt = Ts_gt[frame_start+1:frame_end+1, :, :]
    T_main = np.copy(Ts_gt[0])
    for i in range(0, Ts_gt.shape[0]):
        Tm = np.linalg.inv(T_main).dot(Ts_gt[i,:,:].reshape(4,4))
        Ts_gt[i, :, :] = Tm
    cs_gt = Ts_gt[:, 0:3, 3]
    x_min = np.min(cs_gt[:, 0])
    x_max = np.max(cs_gt[:, 0])
    z_min = np.min(cs_gt[:, 2])
    z_max = np.max(cs_gt[:, 2])

    z_gap = z_max - z_min
    x_gap = x_max - x_min

    w = np.max([x_gap, z_gap])
    z_mid = 0.5*(z_max+z_min)
    z_max = z_mid+w/2
    z_min = z_mid-w/2
    x_mid = 0.5*(x_max+x_min)
    x_max = x_mid+w/2
    x_min = x_mid-w/2


    # T_gt_start = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_start)
    # T_model = np.zeros((3,4))
    # T_model[0:3,0:3] = T_gt_start[0:3,0:3].transpose()
    # T_model[0:3,3] = -T_gt_start[0:3,0:3].transpose().dot(T_gt_start[0:3,3])
    # max_depth = 10
    T_gt = np.zeros((3, 4))
    T_gt[0:3,0:3] = np.eye(3)
    T_dlsu_x_2 = np.zeros((3,4))
    T_dlsu_x_2[0:3, 0:3] = np.eye(3)
    T_ran = np.zeros((3, 4))
    T_ran[0:3, 0:3] = np.eye(3)
    T_epnpu = np.zeros((3, 4))
    T_epnpu[0:3, 0:3] = np.eye(3)
    T_dlsu = np.zeros((3, 4))
    T_dlsu[0:3, 0:3] = np.eye(3)
    fig = plt.figure(figsize=(32, 32))


    ax = fig.add_subplot(111)
    ax.set_xlim(x_min-2, x_max+2)
    # ax.set_ylim3d(-max_depth / 4, max_depth / 4)
    ax.set_ylim(z_min-2, z_max+2)


    Ts_ran = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'ransac_fin')
    Ts_epnpu = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                        'epnpu_fin')
    Ts_dlsu = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                            'dlsu_fin')
    Ts_dlsu_x_2 = reader.read_camera_poses_all('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/',
                                           'dlsu_x_2_fin')

    for frame in range(frame_start, frame_end+1):

        T_ran_m = Ts_ran[frame]
        T_epnpu_m = Ts_epnpu[frame]
        T_dlsu_m = Ts_dlsu[frame]
        T_dlsu_x_2_m = Ts_dlsu_x_2[frame]

        T_gt_m = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt',
                                         frame+1)


        # T_gt[0:3,0:3] = T_gt_m[0:3,0:3].dot(T_gt[0:3,0:3])
        # T_gt[0:3, 3] = T_gt_m[0:3, 0:3].dot(T_gt[0:3, 3]) + T_gt_m[0:3,3]
        T_gt_prev = np.copy(T_gt)
        T_dlsu_x_2_prev = np.copy(T_dlsu_x_2)
        T_ran_prev = np.copy(T_ran)
        T_dlsu_prev = np.copy(T_dlsu)
        T_epnpu_prev = np.copy(T_epnpu)


        T_gt = increment_pose(T_gt_m, T_gt)
        T_dlsu_x_2 = increment_pose(T_dlsu_x_2_m, T_dlsu_x_2)
        T_ran = increment_pose(T_ran_m, T_ran)
        T_epnpu = increment_pose(T_epnpu_m, T_epnpu)
        T_dlsu = increment_pose(T_dlsu_m, T_dlsu)

        T_model = np.eye(4)

        h_epnp = plot_camera_2d(ax, T_ran, T_model, 'b', 'EPnP')
        h_epnpu = plot_camera_2d(ax, T_epnpu, T_model, (0,1,0.0), 'EPnPU')
        # plot_camera_2d(ax, T_dlsu, T_model, (1, 0, 0))
        h_dlsu = plot_camera_2d(ax, T_dlsu_x_2, T_model, (1, 0.0, 0), 'DLSUx2')
        h_gt = plot_camera_2d(ax, T_gt, T_model, (1, 1, 1), 'GT')
        plt.axis('off')
        plt.legend(handles=[h_epnp, h_epnpu, h_dlsu, h_gt], prop={'size': 72})
        plt.savefig('images/video_2d/'+str(frame_start)+ '/' +str(frame)+'.png', dpi=25)
        # plt.close(fig)

        # plt.show()
        # print(frame)

# for i in range(0, 500):
visualize_trajectory_2d(104,500)
    # print(i)
    # visualize_full_trajectory(i, i+20)
# plot_trajs(160, 180)