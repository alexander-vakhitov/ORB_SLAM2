import reader
import skimage.io as io

import skimage.color as skc

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import numpy as np

def plot_camera_2d(ax, T, K, col=(0, 0, 0)):
    s = 0.1
    lw = 2.0
    c = -np.transpose(T[0:3,0:3]).dot(T[0:3, 3])
    cp = K.dot(c)
    cp = cp[0:2]/cp[2]

    corners = []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            corner = np.asarray([s*i,s*j,s])
            corner3d = np.transpose(T[0:3,0:3]).dot(corner-T[0:3, 3])
            corner_proj = K.dot(corner3d)
            corner2d = corner_proj[0:2]/corner_proj[2]
            corners.append(corner2d)

    for i in range(0, 4):
        crn = corners[i]
        ax.plot([cp[0], crn[0]], [cp[1], crn[1]], color=col, linewidth=lw)

    c0 = corners[0]
    c1 = corners[1]
    c2 = corners[2]
    c3 = corners[3]
    ax.plot([c0[0], c1[0]], [c0[1], c1[1]], color=col, linewidth=lw)
    ax.plot([c2[0], c3[0]], [c2[1], c3[1]], color=col, linewidth=lw)
    ax.plot([c3[0], c1[0]], [c3[1], c1[1]], color=col,linewidth=lw)
    ax.plot([c2[0], c0[0]], [c2[1], c0[1]], color=col,linewidth=lw)


def draw_2d_projs(frame_num):
    deb_folder = '/home/alexander/materials/pnp3d/data/debug_pnpu/0/'
    Sigmas = reader.read_sigma_3d(deb_folder, frame_num)
    XX = reader.read_XX(deb_folder, frame_num)
    img_lbl = str(frame_num)
    while len(img_lbl) < 6:
        img_lbl='0'+img_lbl
    img_path = '/home/alexander/materials/sego/kitti_odometry/dataset/sequences/00/image_0/' + img_lbl+'.png'
    img = io.imread(img_path)
    img = skc.gray2rgb(img).astype(float) / 255.0
    T_gt = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_num)

    m = 3
    dls_inliers = reader.read_inliers(deb_folder, frame_num, m)

    epnp_inliers = reader.read_inliers(deb_folder, frame_num, 0)

    xx = reader.read_xx(deb_folder, frame_num)
    sigmas2d = reader.read_sigmas2d(deb_folder, frame_num)

    plt.imshow(img, interpolation='none')
    ax = plt.gca()
    K = np.asarray([[718.856 ,0, 607.193],
      [0, 718.856, 185.216],
      [0 ,      0  ,     1]])

    draw_mode = 1

    for i in range(0, len(xx)):
        if epnp_inliers[i]==1:
            e = Ellipse(xy=xx[i], width=sigmas2d[i], height=sigmas2d[i], angle=0,color=(1,0,0), alpha=0.25)
            ax.add_artist(e)
        Xc = K.dot(T_gt[0:3,0:3].dot(XX[i])+T_gt[0:3,3])
        xc = Xc/Xc[2]
        S3d = K[0,0]*T_gt[0:3,0:3].dot(Sigmas[i]).dot(T_gt[0:3,0:3].transpose())*K[0,0]*1.0/Xc[2]*1.0/Xc[2]
        s2d = S3d[0:2,0:2]
        u,si,v = np.linalg.svd(s2d)
        s = np.sqrt(si)
        d = np.arccos(v[0,1])
        # if draw_mode==2:
        #     e3d = Ellipse(xy = xc[0:2], angle=d*180.0/np.pi, height=s[0], width=s[1], color=(0,0,1), alpha=0.25)
        #     ax.add_artist(e3d)
        if dls_inliers[i]==1:
            s2d = s2d + np.eye(2)*sigmas2d[i]*sigmas2d[i]
            u, si, v = np.linalg.svd(s2d)
            s = np.sqrt(si)
            d = np.arccos(v[0, 1])
            e3d = Ellipse(xy=xx[i], angle=d * 180.0 / np.pi, height=s[0], width=s[1], color=(0, 1, 0), alpha=0.25)
            ax.add_artist(e3d)

    # plt.show()

    T_ransac = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'ransac_fin', frame_num)
    T_x_2 = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'dlsu_x_2_fin',
                                       frame_num)
    T_ep = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'epnpu_fin',
                                    frame_num)
    T_gt = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_num)

    plot_camera_2d(ax, T_gt, K, col=(0, 0, 0))
    plot_camera_2d(ax, T_ransac, K, col=(0, 1, 0))
    plot_camera_2d(ax, T_x_2, K, col=(1, 0, 0))
    plot_camera_2d(ax, T_ep, K, col=(0, 0, 1))

    plt.savefig('images/video/'+str(frame_num)+'.png', bbox_inches='tight', dpi=1000)
    plt.close()
    # plt.show()

for i in range(0,100):
    draw_2d_projs(i)