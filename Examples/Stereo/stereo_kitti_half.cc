/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*
* (C) 2018 Alexander Vakhitov <alexander.vakhitov at gmail dot com>
* Modified: a function main to skip every second frame
*/
#include <include/Sleep.h>

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include <cublas_v2.h>


using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

void runLocalization(const std::string& trajPath, int nCycles, ORB_SLAM2::System& SLAM, const cv::Mat& Two,
        const vector<double>& vTimestamps,
                     const vector<string>& vstrImageLeft, const vector<string>& vstrImageRight)
{
    sleep_ms(10e6);

    std::ofstream relocTrajLog(trajPath + "_traj.txt");
    std::ofstream inlierLog(trajPath + "_inliers.txt");
    std::ofstream baseKFLog(trajPath + "_base_frames.txt");

    for (int ni = 0; ni < nCycles; ni++) {
//        if (ni % 2 == 0) {
//            continue;
//        }

        cv::Mat imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imLeft.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return;
        }

//        std::cout << "F " << ni << std::endl;

        cv::Mat Tcw = SLAM.TrackStereo(imLeft, imRight, tframe);

        int inliers, frameId;
        SLAM.GetLocalizationDetails(&inliers, &frameId);

//        std::cout << " reloc for " << ni << " " << Tcw.cols << " " << Tcw.rows << std::endl;
//        std::cout << " pose: " << std::endl;
//        std::cout << Tcw << std::endl;

        if (Tcw.cols == 0) {
            Tcw = cv::Mat::eye(4, 4, CV_32F);
        } else {
            Tcw = Two.inv() * Tcw.inv();
        }

        for (int ri = 0; ri < 3; ri++) {
            for (int ci = 0; ci < 4; ci++) {
                relocTrajLog << Tcw.at<float>(ri, ci) << " ";
            }
        }
        relocTrajLog << std::endl;

        inlierLog << inliers << std::endl;
        baseKFLog << frameId << std::endl;
    }

    relocTrajLog.close();
}

void runPipeline(const std::string& settingsPath, const std::string& vocPath,
                int trial, int nImages, const vector<double>& vTimestamps,
                const vector<string>& vstrImageLeft, const vector<string>& vstrImageRight, bool is_debug_opt,
                 const std::string& pref, int pnpMode=0)
{
    int n_modes = 2;
    if (is_debug_opt)
    {
        n_modes = 2;
    }
    bool isPoseOpt = true;
    std::string settingsPathMode = settingsPath + "_" + std::to_string(0) + ".yaml";
    ORB_SLAM2::System SLAM(vocPath,settingsPathMode,ORB_SLAM2::System::STEREO,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    bool isCovOpt = (pnpMode>0);

    std::cout << " using pnp mode " << pnpMode << std::endl;

    SLAM.ChangePnpMode(pnpMode, isPoseOpt, isCovOpt, false);


    sleep_ms(10e6);



    // Main loop
    cv::Mat imLeft, imRight;

    int nCycles = nImages;

    for(int ni=0; ni<nCycles; ni++)
    {
        if (ni % 2 == 1)
        {
            continue;
        }

        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        cv::Mat Tr = SLAM.TrackStereo(imLeft,imRight,tframe);
        if (Tr.empty())
        {
            return;
        }


        std::cout << "SLAM " << ni << std::endl;

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            sleep_ms((T-ttrack)*1e6);
    }

    SLAM.ActivateLocalizationMode();

    for (int i = 0; i < 10; i++)
        std::cout << "SLAM STOP" << std::endl;

//    cv::Mat Two = SLAM.SaveTrajectoryKITTI(pref + "/CameraTrajectory_" + std::to_string(trial) + ".txt");
    cv::Mat Two = SLAM.SaveTrajectoryKITTI(pref + "/CameraTrajectory_" + std::to_string(pnpMode) +"_" + std::to_string(trial) + ".txt");

    SLAM.Shutdown();

    return;

    sleep_ms(10e6);

    for (int i = 0; i < 10; i++)
        std::cout << " RELOC START " << nCycles << std::endl;

    SLAM.ChangePnpMode(0, isPoseOpt, false, false);

    runLocalization(pref + "/epnp_" + std::to_string(trial),
            nCycles, SLAM, Two, vTimestamps, vstrImageLeft, vstrImageRight);

    for (int mode = 1; mode < n_modes; mode++) {

        for (int thr_it = 2; thr_it < 3; thr_it ++)
        {
            double thr = 0.5 + thr_it * 0.25;
            SLAM.ChangePnpMode(mode, isPoseOpt, true, false, thr);
            runLocalization(pref + "/epnpu_" + std::to_string(thr_it)+ "_" + std::to_string(trial),
                            nCycles, SLAM, Two, vTimestamps, vstrImageLeft, vstrImageRight);
        }
    }

    // Stop all threads
    SLAM.Shutdown();

//    return 0;
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[3*nImages/4] << endl;
    cout << "mean tracking time: " << totaltime/nImages/2 << endl;

    // Save camera trajectory
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps) {
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for (int i = 0; i < nTimes; i++) {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }

}

void test_pnp_solver()
{
//    std::string debug_path = "/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/pnpu_debug/full_debug_pose.txt";
    std::string debug_path = "/home/alexander/materials/pnp3d/segoexp/debug_solver_cpp/2496/full_debug_pose.txt";
    std::ifstream input_data(debug_path);
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    std::vector<float> sigmas_2d;
    std::vector<float> sigmas_3d;
    std::vector<Eigen::Matrix3d> sigmas_3d_full;
    std::vector<size_t> inds;
    while (!input_data.eof())
    {
        double x;
        input_data >> x;
        cv::Point3f pt_3D;
        pt_3D.x = x;
        input_data >> pt_3D.y >> pt_3D.z;

        cv::Point2f pt_2D;
        input_data >> pt_2D.x >> pt_2D.y;

        float sigma2;
        input_data >> sigma2;

        float sigma3;
        input_data >> sigma3;
        Eigen::Matrix3d S;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double d;
                input_data >> d;
                S(i, j) = d;
            }
        }

        if (input_data.eof())
        {
            break;
        }
        pts_3d.push_back(pt_3D);
        sigmas_2d.push_back(sigma2);
        pts_2d.push_back(pt_2D);
        inds.push_back(inds.size());
        sigmas_3d.push_back(sigma3);
        sigmas_3d_full.push_back(S);

    }

    int mode = 0;
    double fu = 718.856;
    double fv = 718.856;
    double uc = 607.1928;
    double vc = 185.2157;
    std::string debug2_path = "/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/pnpu_debug/debug2/debug_method.txt";
    std::string debug2_pose_path = "/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/pnpu_debug/debug2/debug_pose.txt";
    std::string full_debug2_path = "/home/alexander/materials/pnp3d/segoexp/ORB_SLAM2_SEGO/pnpu_debug/debug2/debug_full.txt";
    ORB_SLAM2::PnPsolver pSolver(pts_2d, sigmas_2d, pts_3d, inds, inds, fu, fv, uc, vc, inds.size(), sigmas_3d,
                                 sigmas_3d_full, mode, false, 1.0, true, debug2_path, debug2_pose_path, full_debug2_path);
    pSolver.SetRansacParameters(0.99, 5, 300, 4, 0.1, 5.991);
    int nInliers;
    bool bNoMore;
    std::vector<bool> vbInliers;
    std::cout << " starting iterations " << pts_2d.size() << std::endl;
    cv::Mat Tcw = pSolver.iterate(5,bNoMore,vbInliers,nInliers);

    std::cout << " got " << Tcw << " inliers " << nInliers << std::endl;

    if (Tcw.empty())
    {
        pSolver.SetRansacParameters(0.99, 5, 300, 4, 0.01, 5.991);
        cv::Mat Tcw = pSolver.iterate(5,bNoMore,vbInliers,nInliers);
        std::cout << " got " << Tcw << " inliers " << nInliers << std::endl;
    }
}


int main(int argc, char **argv)
{

    test_pnp_solver();
    return 0;

    if(argc < 6)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    int pnpMode = 0;
    if (argc >= 7)
    {
        pnpMode = atoi(argv[6]);
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::string settingsPath = argv[2];

//    int mode = atoi(argv[4]);
//    int trial = atoi(argv[5]);

    int trial = atoi(argv[4]);

    bool is_debug_opt = false;

    runPipeline(settingsPath, argv[1], trial, nImages, vTimestamps, vstrImageLeft, vstrImageRight, is_debug_opt, argv[5],
                pnpMode);

//    for (int mode = 0; mode < 4; mode++) {
//        for (int trial = 0; trial < 5; trial++)
//        {

//        }

//    }

    return 0;
}

