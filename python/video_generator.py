import skimage.io as skio
import skimage.transform as skt
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_kitti_image(fpath):
    left = 116
    right = 1547
    top = 37
    bottom = 470
    im1 = cv2.imread(fpath)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
    im1 = im1[top:bottom, left:right, :]
    # im1 = skt.rescale(im1, 0.25)
    return im1

def read_map_image(fpath):
    im_map = cv2.imread(fpath)
    # im_map = cv2.cvtColor(im_map, cv2.COLOR_RGB2BGR)
    im_map = skt.rescale(im_map, 0.5)*255
    return im_map

def read_map3d_image(fpath):
    im_map = cv2.imread(fpath)
    # im_map = cv2.cvtColor(im_map, cv2.COLOR_RGB2BGR)
    im_map = im_map[:, 220:1080, :]
    # im_map = skt.rescale(im_map, 0.5)
    return im_map

def generate_figure(frame_num, is_plain):
    kitti_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/tracking/'
    map_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/video_2d/'
    map3d_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/map/'
    video_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/finvideo/'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    leg_det = cv2.imread('/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/legend_detector.png')
    leg_det = skt.rescale(leg_det, 0.2) * 255
    leg_frust = cv2.imread('/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/legend_frustum.png')
    leg_frust = skt.rescale(leg_frust, 0.5) * 255
    im1 = read_kitti_image(kitti_path + '/'+str(frame_num)+'_2.png')
    h, w = im1.shape[0:2]
    if is_plain:
        im1 = read_kitti_image(kitti_path + '/' +str(frame_num)+'_1.png')

    im_map3d = read_map3d_image(map3d_path + str(frame_num) + '.png')
    if is_plain:
        im_map3d = read_map3d_image(map3d_path + str(frame_num) + '_plain_.png')

    # h_map, w_map = im_map.shape[0:2]
    # h,w = im1.shape[0:2]
    big_im = np.zeros((2 * h, w, 3), dtype=np.uint8)
    big_im[0:h, :, :] = im1[:, :, 0:3].astype(np.uint8)
    # big_im[h:2*h, :, :] = im2[:,:,0:3].astype(np.uint8)
    # mapy = int(h + h / 2 - h_map / 2)
    # mapx = w - w_map
    # big_im[mapy:mapy + h_map, mapx:mapx + w_map, :] = im_map[:, :, 0:3].astype(np.uint8)

    h_map3d, w_map3d = im_map3d.shape[0:2]

    map3dy = int(h + h / 2 - h_map3d / 2)
    w_mid = w / 2
    w_map_start = int(w_mid - im_map3d.shape[1] / 2)
    big_im[h:h + h_map3d, w_map_start:w_map_start+im_map3d.shape[1], :] = im_map3d[:, :, 0:3].astype(np.uint8)

    if not is_plain:
        w_mid = w / 2
        w_leg = leg_det.shape[1]
        big_im[h:h + leg_det.shape[0], int(w- w_leg) :w,:] = leg_det



    big_im[h +50: h + 50+leg_frust.shape[0],100:100+ leg_frust.shape[1], :] = leg_frust
    # big_im = cv2.cvtColor(big_im, cv2.COLOR_RGB2BGR)
    print(big_im.shape)
    plt.imshow(big_im)
    plt.show()
    cv2.imwrite('images/ill.png', big_im)


def generate_video(frame_start, frame_end):
    kitti_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/tracking/'
    map_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/video_2d/'
    map3d_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/map/'
    video_path = '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/finvideo/'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    leg_det = cv2.imread('/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/legend_detector.png')
    leg_det = skt.rescale(leg_det, 0.2) * 255
    leg_frust = cv2.imread('/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/python/images/legend_frustum.png')
    leg_frust = skt.rescale(leg_frust, 0.5) * 255
    im1 = read_kitti_image(kitti_path + '/1_2.png')
    h, w = im1.shape[0:2]
    vw = cv2.VideoWriter(video_path+str(frame_start)+'.avi', fourcc, 20, (w,2*h))
    print((w,2*h))
    for frame_num in range(frame_start, frame_end+1):
        im1 = read_kitti_image(kitti_path + '/' + str(frame_num) + '_2.png')
        # im2 = read_kitti_image(kitti_path + '/' + str(frame_num) + '_2.png')
        # start_frame = 20*int(frame_num/20)
        im_map = read_map_image(map_path+'/104/'+str(frame_num)+'.png')
        # plt.imshow(im_map)
        # plt.show()

        im_map3d = read_map3d_image(map3d_path+str(frame_num)+'.png')

        h_map, w_map = im_map.shape[0:2]
        # h,w = im1.shape[0:2]
        big_im = np.zeros((2*h, w, 3), dtype=np.uint8)
        big_im[0:h, :, :] = im1[:,:,0:3].astype(np.uint8)
        # big_im[h:2*h, :, :] = im2[:,:,0:3].astype(np.uint8)
        mapy = int(h+h/2-h_map/2)
        mapx = w - w_map
        big_im[mapy:mapy+h_map, mapx:mapx+w_map, :] = im_map[:,:,0:3].astype(np.uint8)

        h_map3d, w_map3d = im_map3d.shape[0:2]

        map3dy = int(h+h/2-h_map3d/2)
        big_im[map3dy:map3dy + h_map3d, 0:w_map3d, :] = im_map3d[:, :, 0:3].astype(np.uint8)

        w_mid = w/2
        w_leg = leg_det.shape[1]
        big_im[h:h+leg_det.shape[0], int(w_mid-w_leg/2)+50:50+int(w_mid-w_leg/2)+leg_det.shape[1], :] = leg_det

        big_im[2*h-leg_frust.shape[0]: 2*h, 0:leg_frust.shape[1], :] = leg_frust
        # big_im = cv2.cvtColor(big_im, cv2.COLOR_RGB2BGR)
        vw.write(big_im)
        print(big_im.shape)
        # plt.imshow(big_im)
        # plt.show()


generate_video(104, 500)
# generate_figure(104, False)