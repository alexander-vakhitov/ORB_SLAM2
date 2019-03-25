import reader
import skimage.io as io

import skimage.color as skc

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import numpy as np

def draw_2d_projs(frame_num):
    deb_folder = '/home/alexander/materials/pnp3d/data/debug_pnpu/0/'
    Sigmas = reader.read_sigma_3d(deb_folder, frame_num)
    XX = reader.read_XX(deb_folder, frame_num)
    img_path = '/home/alexander/materials/sego/kitti_odometry/dataset/sequences/00/image_0/00' + str(frame_num)+'.png'
    img = io.imread(img_path)
    img = skc.gray2rgb(img).astype(float) / 255.0
    T_gt = reader.read_gt_pose('/home/alexander/materials/sego/kitti_odometry/dataset/poses/00.txt', frame_num)

    m = 3
    dlus_inliers = reader.read_inliers(deb_folder, frame_num, m)
    xx = reader.read_xx(deb_folder, frame_num)
    sigmas2d = reader.read_sigmas2d(deb_folder, frame_num)

    print(np.max(sigmas2d))
    print(np.min(sigmas2d))

    plt.imshow(img, interpolation='none')

    ax = plt.gca()

    K = np.asarray([[718.856 ,0, 607.193],
      [0, 718.856, 185.216],
      [0 ,      0  ,     1]])

    draw_mode = 1

    for i in range(0, len(xx)):
        if dlus_inliers[i] == 0:
            continue
        if draw_mode==1:
            e = Ellipse(xy=xx[i], width=sigmas2d[i], height=sigmas2d[i], angle=0,color=(1,0,0), alpha=0.25)
            ax.add_artist(e)
        Xc = K.dot(T_gt[0:3,0:3].dot(XX[i])+T_gt[0:3,3])
        xc = Xc/Xc[2]
        S3d = K[0,0]*T_gt[0:3,0:3].dot(Sigmas[i]).dot(T_gt[0:3,0:3].transpose())*K[0,0]*1.0/Xc[2]*1.0/Xc[2]
        s2d = S3d[0:2,0:2]
        u,si,v = np.linalg.svd(s2d)
        s = np.sqrt(si)
        print(u.dot(np.diag(si)).dot(v)-s2d)
        d = np.arccos(v[0,1])
        print(np.arccos(0))
        if draw_mode==2:
            e3d = Ellipse(xy = xc[0:2], angle=d*180.0/np.pi, height=s[0], width=s[1], color=(0,0,1), alpha=0.25)
            ax.add_artist(e3d)
        if draw_mode == 3:
            s2d = s2d + np.eye(2)*sigmas2d[i]*sigmas2d[i]
            u, si, v = np.linalg.svd(s2d)
            s = np.sqrt(si)
            d = np.arccos(v[0, 1])
            e3d = Ellipse(xy=xc[0:2], angle=d * 180.0 / np.pi, height=s[0], width=s[1], color=(0, 1, 0), alpha=0.25)
            ax.add_artist(e3d)

    # plt.show()
    plt.savefig('images/'+str(draw_mode)+'.pdf', bbox_inches='tight', dpi=1000)


draw_2d_projs(2804)