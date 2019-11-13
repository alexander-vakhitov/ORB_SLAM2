import reader
import skimage.io as io

import skimage.color as skc

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from matplotlib import cm
import matplotlib.colors as colors

import numpy as np

import ellipse_proj_test as ep

def rep_err(T, K, X, x):
    Xc = K.dot(T[0:3,0:3].dot(X) + T[0:3,3])
    xc = Xc[0:2] / Xc[2]
    return np.linalg.norm(x-xc)

def calc_rep_errs(XX,xx,T,K,inliers):
    n = len(inliers)
    rep_errs = np.zeros(n)
    rep_err_lst = []
    for i in range(0, n):
        if inliers[i] == 1:
            rep_errs[i] = rep_err(T, K, XX[i], xx[i])
            rep_err_lst.append(rep_errs[i])
    rep_err_lst = sorted(rep_err_lst)
    n = len(rep_err_lst)
    max_err = rep_err_lst[int(0.95*n)]
    min_err = rep_err_lst[int(0.05 * n)]
    return rep_errs, max_err, min_err


def draw_2d_projs(frame_num, draw_mode):
    # plt.style.use('dark_background')
    deb_folder = '/home/alexander/materials/pnp3d/data/debug_pnpu/0/'

    K = np.asarray([[718.856, 0, 607.193],
                    [0, 718.856, 185.216],
                    [0, 0, 1]])

    Sigmas = reader.read_sigma_3d(deb_folder, frame_num)
    XX = reader.read_XX(deb_folder, frame_num)
    xx = reader.read_xx(deb_folder, frame_num)
    sigmas2d = reader.read_sigmas2d(deb_folder, frame_num)
    img_lbl = str(frame_num+1)
    while len(img_lbl)<6:
        img_lbl = '0'+img_lbl
    img_path = '/home/alexander/materials/sego/kitti_odometry/dataset/sequences/00/image_0/' + img_lbl +'.png'
    img = io.imread(img_path)
    img = skc.gray2rgb(img).astype(float) / 255.0
    T_gt = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_num+1)
    T_ran = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'ransac_fin',
                                    frame_num)
    T_dlsu = reader.read_camera_pose('/home/alexander/materials/pnp3d/data/test_line_verification/0/0/', 'dlsu_x_2_fin',
                                    frame_num)

    m = 3
    dlsu_inliers = reader.read_inliers(deb_folder, frame_num, m)

    ransac_inliers = reader.read_inliers(deb_folder, frame_num, 0)

    ransac_rep_errs, max_ransac, min_ransac = calc_rep_errs(XX, xx, T_ran, K, ransac_inliers)
    dlsu_rep_errs, max_dlsu, min_dlsu = calc_rep_errs(XX, xx, T_dlsu, K, dlsu_inliers)

    norm = colors.Normalize(vmin=0, vmax=np.max(sigmas2d))
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)



    plt.imshow(img, interpolation='none')

    ax = plt.gca()

    unc_traces = []
    for i in range(0, len(xx)):
        if dlsu_inliers[i] == 1:
            Sigma3dCam = T_dlsu[0:3, 0:3].dot(np.linalg.inv(Sigmas[i])).dot(T_dlsu[0:3, 0:3].transpose())
            c_cam = T_dlsu[0:3, 0:3].dot(XX[i]) + T_dlsu[0:3, 3]
            S2d, c2d, d2d = ep.project_ellipse(Sigma3dCam, K, c_cam, 1)
            S2d = S2d + np.eye(2) * 1.0 / (sigmas2d[i] * sigmas2d[i])
            unc_traces.append(np.sqrt(np.trace(S2d)))

    unc_traces = sorted(unc_traces)
    n_unc = len(unc_traces)
    min_unc = unc_traces[int(0.05*n_unc)]
    max_unc = unc_traces[int(0.95 * n_unc)]

    # draw_mode = 1

    # colors = []

    ms = []

    for i in range(0, len(xx)):
        # if dlsu_inliers[i] == 1 or ransac_inliers [i] == 1:
        #     plt.plot(xx[i][0], xx[i][1], 'g+')

        if draw_mode == 1 and ransac_inliers [i] == 1:
            # e = Ellipse(xy=xx[i], width=sigmas2d[i], height=sigmas2d[i], angle=0,color=(1,0,0), alpha=0.25)
            # ax.add_artist(e)
            Xc = K.dot(T_ran[0:3, 0:3].dot(XX[i]) + T_ran[0:3, 3])
            xc = Xc / Xc[2]

            err = ransac_rep_errs[i]
            err_norm = (err - min_ransac)/(max_ransac-min_ransac)
            print(cmap.N)
            color_val = cmap(int(err_norm * (cmap.N-1)))
            print(err_norm)
            ep.draw_ellipse(ax, np.eye(2), xx[i], 1, col=(0,0,1), alpha=1.0)
            # dx = xx[i]-xc[0:2]
            # plt.plot([xc[0], xc[0]+5*dx[0]], [xc[1], xc[1]+5*dx[1]], color=(1,0,0), alpha=0.5, linewidth=2.0)


        # continue
            # plt.plot(xc[0], xc[1], 'm+')
            # plt.plot([xc[0],xx[i][0]], [xc[1],xx[i][1]], 'm')
        # Xc = K.dot(T_gt[0:3,0:3].dot(XX[i])+T_gt[0:3,3])
        # xc = Xc/Xc[2]
        # S3d = K[0,0]*T_gt[0:3,0:3].dot(Sigmas[i]).dot(T_gt[0:3,0:3].transpose())*K[0,0]*1.0/Xc[2]*1.0/Xc[2]
        # s2d = S3d[0:2,0:2]
        # u,si,v = np.linalg.svd(s2d)
        # s = np.sqrt(si)
        # print(u.dot(np.diag(si)).dot(v)-s2d)
        # d = np.arccos(v[0,1])
        # print(np.arccos(0))
        # c_cam = T_dlsu[0:3, 0:3].dot(XX[i]) + T_dlsu[0:3, 3]
        # S2d, c2d, d2d = ep.project_ellipse(Sigma3dCam, K, c_cam, 1)


        # if draw_mode==2:
        #     if d2d>0 and c_cam[2]>1 and np.linalg.norm(c_cam)<100:
        #         print(c2d-xx[i])
        #         ep.draw_ellipse(ax, S2d, xx[i], d2d, col=(0,0,1), alpha=0.25)
        color_val = (0,0,0,1)
        if (draw_mode == 2 or draw_mode == 3) and dlsu_inliers[i] == 1:
            Sigma3dCam = T_dlsu[0:3, 0:3].dot(Sigmas[i]).dot(T_dlsu[0:3, 0:3].transpose())
            c_cam = T_dlsu[0:3,0:3].dot(XX[i]) + T_dlsu[0:3,3]
            SigmaInv = np.linalg.inv(Sigma3dCam)
            S2d, c2d, d2d = ep.project_uncertainty_slice(SigmaInv, K, c_cam)
            # if (np.trace(S2d) < 0):
            #     print(Sigma3dCam)
            #     print(c_cam)
            #     print(K)
            #     S2d
            if draw_mode==2:
                S2d = np.linalg.inv(np.eye(2)*(sigmas2d[i]*sigmas2d[i]) + np.linalg.inv(S2d))
            else:
                S2d = np.linalg.inv(np.eye(2) * (sigmas2d[i] * sigmas2d[i]))
            # np.trace(S2d)

            ec = ep.draw_ellipse(ax, S2d, xx[i], 1, col=m.to_rgba(sigmas2d[i]), alpha=1.0)
            # ms.append(ec)
            # ep.draw_ellipse(ax, S2d2d, xx[i], 1, col=(0,0,1), alpha=1.0)


            # ep.draw_ellipse(ax, np.eye(2) , xx[i], 4.0, col=color_val, alpha=0.5)
            # ep.draw_ellipse(ax, s2d, c2d, 10*d2d, col=(0, 0, 1), alpha=0.25)
            # plt.plot(c2d[0], c2d[1], 'r+')
            # plt.plot([c2d[0], xx[i][0]], [c2d[1],xx[i][1]], 'r')


            # u, si, v = np.linalg.svd(s2d)
            # s = np.sqrt(si)
            # d = np.arccos(v[0, 1])
            # e3d = Ellipse(xy=xx[i], angle=d * 180.0 / np.pi, height=s[0], width=s[1], color=(0, 1, 0), alpha=0.25)
            # ax.add_artist(e3d)
        # colors.append(color_val)

    plt.savefig('images/tracking_paper/'+str(frame_num)+'.png')
    plt.close()

def detector_legend():
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use('dark_background')
    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    # ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    norm = colors.Normalize(vmin=0, vmax=8.1)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Detector uncertainty', size =20)

    # legend
    # cbar = plt.colorbar(heatmap)

    plt.axis('off')
    plt.gcf().tight_layout()
    # plt.show()


    plt.savefig('images/legend_detector.png', bbox_inches='tight', dpi=250)
    # plt.close()
    return []

# for frame_unm in range(104, 500):
#     for draw_mode in range(2, 3):
#         draw_2d_projs(frame_unm, draw_mode)

# detector_legend()

draw_2d_projs(119, 2)
draw_2d_projs(157, 2)
draw_2d_projs(227, 2)
draw_2d_projs(275, 2)