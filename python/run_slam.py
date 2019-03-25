import subprocess
import os
import shutil

def run_slam():
    for mode_id in range(0, 4):
        store_dir = 'RELOC_mode_'+str(mode_id)
        if not os.path.exists(store_dir ):
            os.makedirs(store_dir )
        mode_config_file = 'KITTI00-02_'+str(mode_id)+'.yaml'
        for ti in range(0, 5):
                subprocess.call(['Examples/Stereo/stereo_kitti_half', 'Vocabulary/ORBvoc.txt',
                                         '/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/Examples/Stereo/'+mode_config_file,
                                         '/home/alexander/materials/sego/kitti_odometry/dataset/sequences/00'])

                shutil.move('RelocalizationTrajectory.txt', store_dir+'/traj_00_'+str(ti)+'.txt')


run_slam()