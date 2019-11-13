import reader
import eldrawer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
import matplotlib.colors as colors

import simple_ellipsoid

from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png

from pylab import *

import skimage.io as io
import skimage.color as skc
import skimage.transform as skt

def plot_camera(ax, T, col=(0, 0, 0), lw = 1.0):
    s = 4*0.3
    # lw = 1.0
    c = -np.transpose(T[0:3,0:3]).dot(T[0:3, 3])

    corners = []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            corner = np.asarray([s*i,s*j,s])
            corners.append(np.transpose(T[0:3,0:3]).dot(corner-T[0:3, 3]))

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



def draw_map_covar(frame_num, draw_ellipses = True, is_draw_surf = False):
    # plt.style.use('dark_background')
    deb_folder = '/home/alexander/materials/pnp3d/data/debug_pnpu/0/'
    Sigmas = reader.read_sigma_3d(deb_folder, frame_num)
    XX = reader.read_XX(deb_folder, frame_num)
    sigmas = reader.read_sigmas2d(deb_folder, frame_num)
    inliers = reader.read_inliers(deb_folder, frame_num, 3)
    rinliers = reader.read_inliers(deb_folder, frame_num, 0)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111, projection='3d')

    # number of ellipsoids
    ellipNumber = len(XX)

    # set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=0, vmax=np.max(sigmas))
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    max_depth = 30

    Rv = np.eye(3)
    Rv[2,2] = 0
    Rv[1,1] = 0
    Rv[2,1] = -1
    Rv[1,2] = 1

    T_ransac = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'ransac_fin', frame_num)
    T_x_2 = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'dlsu_x_2_fin',
                                       frame_num)
    T_ep = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'epnpu_fin',
                                    frame_num)
    T_gt = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_num)
    print(T_gt)
    Rve = np.eye(4)
    Rve[0:3,0:3] = Rv


    if draw_ellipses:
        for cnt in range(0, len(XX)):
            if inliers[cnt] == 0:
                continue
            # if np.linalg.norm(XX[cnt]) > 300:
            #     continue
            U2, s2, rotation2 = np.linalg.svd(Sigmas[cnt])
            if np.mean(np.sqrt(s2)) > max_depth:
                continue
            if XX[cnt][2]<0:
                continue
            if XX[cnt][2]>max_depth:
                continue

            s2d = sigmas[cnt]
            x,y,z = eldrawer.define_ellipsoid(Rv.dot(XX[cnt]), Rv.dot(Sigmas[cnt]).dot(Rv.transpose()))
            # x, y, z = eldrawer.define_ellipsoid([0,0,0], np.eye(3), 1.0)
            #ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(cnt), linewidth=0.1, alpha=1, shade=True)
            ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=m.to_rgba(s2d), linewidth=0.1, alpha=1)
        plot_camera(ax, T_gt.dot(Rve.transpose()), col=(0, 0, 0), lw=4.0)
        plot_camera(ax, T_ransac.dot(Rve.transpose()), col=(0, 0, 1), lw=3.0)
        plot_camera(ax, T_x_2.dot(Rve.transpose()), col=(1, 0, 0), lw=2.0)
        plot_camera(ax, T_ep.dot(Rve.transpose()), col=(0, 1, 0))

    else:

        for cnt in range(0, len(XX)):
            if rinliers[cnt] == 0:
                continue
            # if np.linalg.norm(XX[cnt]) > 300:
            #     continue
            if XX[cnt][2] < 0:
                continue
            if XX[cnt][2] > max_depth:
                continue

            s2d = sigmas[cnt]
            x, y, z = eldrawer.define_ellipsoid(Rv.dot(XX[cnt]), np.eye(3)*0.01)
            # x, y, z = eldrawer.define_ellipsoid([0,0,0], np.eye(3), 1.0)
            # ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(cnt), linewidth=0.1, alpha=1, shade=True)
            ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=(0,0,1), linewidth=0.1, alpha=1)

        plot_camera(ax, T_gt.dot(Rve.transpose()), col=(0, 0, 0), lw=4.0)
        plot_camera(ax, T_ransac.dot(Rve.transpose()), col=(0, 0, 1), lw=3.0)

    #plot camera

    # ax.set_xlim3d(-max_depth / 2, max_depth / 2)
    ax.set_xlim3d(-max_depth/3, max_depth / 3)
    #ax.set_ylim3d(-max_depth / 4, max_depth / 4)
    # ax.set_ylim3d(-max_depth/3, max_depth / 3)
    # ax.set_zlim3d(0, max_depth)
    ax.set_ylim3d(0, max_depth)
    ax.set_zlim3d(-max_depth / 10, max_depth / 3)


    # plot_camera(ax, np.eye(4), col=(0, 0, 0))


    if is_draw_surf:
        img = io.imread('/home/alexander/materials/sego/kitti_odometry/dataset/sequences/00/image_0/00' + str(frame_num)+'.png')
        img = skc.gray2rgb(img).astype(float)/255.0
        img = skt.rescale(img, 0.1)

        x = np.outer(np.ones(img.shape[0]), np.linspace(-max_depth/2, max_depth/2, img.shape[1]))
        y = np.outer(np.linspace(-max_depth / 2, max_depth / 2, img.shape[0]), np.ones(img.shape[1]))
        ax.plot_surface(x, y, max_depth*np.ones_like(x), rstride=1, cstride=1, facecolors=img, alpha=0.2)

    ax.view_init(elev=50,azim=-30)
    # plt.show()
    # simple_ellipsoid.draw_ellipsoids(XX, Sigmas)
    suff = ''
    if not draw_ellipses:
        suff = '_plain_'
    plt.savefig('images/map_paper/'+str(frame_num)+suff+'.png')

def make_legend():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(3, 4))
    ax = fig.add_subplot(111, projection='3d')

    Rv = np.eye(3)
    Rv[2,2] = 0
    Rv[1,1] = 0
    Rv[2,1] = -1
    Rv[1,2] = 1

    Rve = np.eye(4)
    Rve[0:3,0:3] = Rv

    T0 = np.eye(4).dot(Rve.transpose())
    T1 = np.copy(T0)
    T1[1,3] = 4

    T2 = np.copy(T0)
    T2[1, 3] = 8

    T3 = np.copy(T0)
    T3[1, 3] = 12

    plot_camera(ax, T0, col=(0, 0, 0), lw=4.0)
    plot_camera(ax, T1, col=(0, 0, 1), lw=3.0)
    plot_camera(ax, T2, col=(1, 0, 0), lw=2.0)
    plot_camera(ax, T3, col=(0, 1, 0))
    ax.grid(False)
    ax.set_xlim3d(-15,15)
    ax.set_ylim3d(-8, 8)
    ax.set_zlim3d(-15, 15)

    # plt.show()
    plt.savefig('images/legend_frustum.png', dpi=250)

# for i in range(104, 500):
#     draw_map_covar(i, False)

# make_legend()

draw_map_covar(119)
draw_map_covar(157)
draw_map_covar(227)
draw_map_covar(275)