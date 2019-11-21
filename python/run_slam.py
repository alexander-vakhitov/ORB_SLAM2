import subprocess
import os
import shutil
import sys
import time

def run_slam(seq_id='00', is_debug_opt=False):

    for ti in range(0, 3):
        store_dir = 'kittipnpu/' + seq_id + '/'
        if not os.path.exists(store_dir ):
            os.makedirs(store_dir )
        mode_config_file = 'KITTI00-02'#+str(mode_id)+'.yaml'
        fname_test = 'CameraTrajectory_' + str(ti) + '.txt'
        is_debug_opt_flag = 0
        if is_debug_opt:
            is_debug_opt_flag = 1
        subprocess.call([
            'Examples/Stereo/stereo_kitti_half', 'Vocabulary/ORBvoc.txt',
            '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/Examples/Stereo/'+mode_config_file,
            '/home/alexander/materials/sego/kitti_odometry/dataset/sequences/'+str(seq_id),
            str(ti),
            store_dir])
        # fname = 'CameraTrajectory_' + str(ti) + '.txt'
        # if os.path.exists(fname):
        #     shutil.move(fname, store_dir + '/' + fname)
        # for mode_id in range(0, 4):
        #     fname = 'RelocalizationTrajectory' + str(mode_id) + '_' + str(ti) + '.txt'
        #     if os.path.exists(fname):
        #         shutil.move(fname, store_dir + '/' + fname)


def run_slam_half(seq_id='00', is_debug_opt=False):

    for ti in range(0, 5):
        for mi in range(0, 2):
            store_dir = 'kittipnpu/' + seq_id + '/'
            if not os.path.exists(store_dir ):
                os.makedirs(store_dir )
            mode_config_file = 'KITTI00-02'#+str(mode_id)+'.yaml'
            is_debug_opt_flag = 0
            if is_debug_opt:
                is_debug_opt_flag = 1
            subprocess.call([
                'Examples/Stereo/stereo_kitti_half', 'Vocabulary/ORBvoc.txt',
                '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/Examples/Stereo/'+mode_config_file,
                '/home/alexander/materials/sego/kitti_odometry/dataset/sequences/'+str(seq_id),
                str(ti),
                store_dir,
                str(mi)])

def run_kitti_half(seq_id, q=1, is_debug_opt=False):

    seq_lbl = str(seq_id)
    if len(seq_lbl) < 2:
        seq_lbl = '0' + seq_lbl

    for ti in range(0, 3):
        for mi in [0,3]:
            store_dir = 'kittipnpudemo_' +str(q)+"/"+ seq_lbl + '/'
            if not os.path.exists(store_dir ):
                os.makedirs(store_dir )
            img_dir = store_dir + '/' + str(mi) + '_' + str(ti)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            mode_config_file = 'KITTI00-02'#+str(mode_id)+'.yaml'
            if seq_id == 3:
                mode_config_file = 'KITTI03'
            if seq_id > 3:
                mode_config_file = 'KITTI04-12'
            is_debug_opt_flag = 0
            if is_debug_opt:
                is_debug_opt_flag = 1

            subprocess.call([
                'Examples/Stereo/stereo_kitti_half', 'Vocabulary/ORBvoc.txt',
                'Examples/Stereo/'+mode_config_file,
                '/storage/kitti/dataset/sequences/'+seq_lbl,
                str(ti),
                store_dir,
                str(mi),
                str(q)])

def run_slam_euroc(seq_id=0):
    seq_lbls = ['MH_01_easy', 'MH_02_easy', 'MH_03_medium']
    ts_lbls = ['MH01', 'MH02', 'MH03']
    slbl = seq_lbls[seq_id]
    tslbl = ts_lbls[seq_id]
    for ti in range(0, 3):
        # for mode_id in range(0, 4):
        store_dir = "pnpueuroc" + '/' + slbl + '/'
        if not os.path.exists(store_dir ):
            os.makedirs(store_dir )
        # while not os.path.exists(fname_test):
        subprocess.call([
            'Examples/Stereo/stereo_euroc_half', 'Vocabulary/ORBvoc.txt',
            '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/Examples/Stereo/EuRoC.yaml',
            '/home/alexander/materials/line_descriptor/euroc/{}/mav0/cam0/data/'.format(slbl),
            '/home/alexander/materials/line_descriptor/euroc/{}/mav0/cam1/data/'.format(slbl),
            '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/Examples/Stereo/EuRoC_TimeStamps/{}.txt'.format(tslbl),
            str(ti),
            store_dir])
        # fname = 'CameraTrajectory_' + str(ti) + '.txt'
        # if os.path.exists(fname):
        #     shutil.move(fname, store_dir + '/' + fname)
        # for mode_id in range(0, 4):
        #     fname = 'RelocalizationTrajectory' + str(mode_id) + '_' + str(ti) + '.txt'
        #     if os.path.exists(fname):
        #         shutil.move(fname, store_dir + '/' + fname)


def run_slam_euroc_half(seq_id=0, q=1):
    seq_lbls = ['MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult', 'V1_01_easy', 'V1_02_medium',
                'V1_03_difficult', 'V2_01_easy', 'V2_02_medium', 'V2_03_difficult']
    ts_lbls = ['MH01', 'MH02', 'MH03', 'MH04', 'MH05', 'V101', 'V102', 'V103', 'V201', 'V202', 'V203']
    slbl = seq_lbls[seq_id]
    tslbl = ts_lbls[seq_id]
    for ti in range(0, 5):
        #for mi in range(0, 4):
        for mi in [0,3]:
            # for mode_id in range(0, 4):
            store_dir = "pnpueuroc_night2_check_" +str(q) + '/' + slbl + '/'
            if not os.path.exists(store_dir ):
                os.makedirs(store_dir )
            # while not os.path.exists(fname_test):
            command_lst = [
                'Examples/Stereo/stereo_euroc_half', 'Vocabulary/ORBvoc.txt',
                'Examples/Stereo/EuRoC.yaml',
                '/storage/lld/lld_code/euroc/{}/mav0/cam0/data/'.format(slbl),
                '/storage/lld/lld_code/euroc/{}/mav0/cam1/data/'.format(slbl),
                'Examples/Stereo/EuRoC_TimeStamps/{}.txt'.format(tslbl),
                str(ti),
                store_dir,
                str(mi),
                str(q)]


            s = ''
            for l in command_lst:
                s = s + l + ' '
            print(s)

            subprocess.call(command_lst)

            time.sleep(5)

            # x = input("check traj!")
        # fname = 'CameraTrajectory_' + str(ti) + '.txt'
        # if os.path.exists(fname):
        #     shutil.move(fname, store_dir + '/' + fname)
        # for mode_id in range(0, 4):
        #     fname = 'RelocalizationTrajectory' + str(mode_id) + '_' + str(ti) + '.txt'
        #     if os.path.exists(fname):
        #         shutil.move(fname, store_dir + '/' + fname)


if __name__ == '__main__':
    # run_slam_euroc_half(0)
    # run_slam_euroc(2)
    # run_slam_euroc(0)
    # run_slam_euroc( 1)

    # run_slam('', '00')
    # run_slam(sys.argv[1], '01')
    # run_slam(sys.argv[1], '03')

    # run_slam('00')
    # run_slam('01')
    # run_slam('02')
    # run_slam_half('00')
    # run_slam_half('01')
    # run_slam_half('02')
    for q in range(2, 3):
        for si in range(1, 2):
            # run_slam_euroc_half(si, q)
            run_kitti_half(si, q)
    # run_slam_euroc_half(1)
    # run_slam_euroc_half(2)