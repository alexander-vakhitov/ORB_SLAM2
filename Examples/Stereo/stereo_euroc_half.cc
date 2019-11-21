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
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include <Eigen/Dense>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "Sleep.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>
#include <opencv/cxeigen.hpp>

#include "Converter.h"


using namespace std;

using namespace ORB_SLAM2;

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void runLocalization(const std::string& trajPath, int nCycles, ORB_SLAM2::System& SLAM, const cv::Mat& Two,
                     const vector<double>& vTimestamps,
                     const vector<string>& vstrImageLeft, const vector<string>& vstrImageRight,
                     const cv::Mat& M1l, const cv::Mat& M2l, const cv::Mat& M1r, const cv::Mat& M2r)
{
    sleep_ms(10e6);

    std::ofstream relocTrajLog(trajPath + "_traj.txt");
    std::ofstream inlierLog(trajPath + "_inliers.txt");
    std::ofstream baseKFLog(trajPath + "_base_frames.txt");

    for (int ni = 0; ni < nCycles; ni++) {
        if (ni % 3 == 0) {
            continue;
        }

        cv::Mat imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imLeft.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return;
        }

//        std::cout << "F " << ni << std::endl;

        cv::Mat imLeftRect, imRightRect;

        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

        cv::Mat Tcw = SLAM.TrackStereo(imLeftRect,imRightRect,tframe);

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

void test_refinement()
{
    int covUseMode = 1;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

    Eigen::Vector3d t;
    t.setRandom();
    t(2) += 5;

    Eigen::Vector3d r;
    r.setRandom();

    cv::Mat rvec(3, 1, CV_32FC1);
    for (int i = 0; i < 3; i++)
    {
        rvec.at<float>(i, 0) = r(i);
    }
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    Eigen::Matrix3d R_eig;
    cv::cv2eigen(R_cv, R_eig);

    cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
    T.rowRange(0, 3).colRange(0, 3) = R_cv;
    for (int i = 0; i < 3; i++)
    {
        T.at<float>(i, 3) = t(i);
    }

    g2o::SE3Quat se3_init = Converter::toSE3Quat(T);
    vSE3->setEstimate(se3_init);
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = 20;

    vector<g2o::EdgeSE3CovProjectXYZOnlyPose*> vpEdgesMono;
//    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3CovProjectXYZOnlyPose*> vpEdgesStereo;
//    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    Eigen::Matrix3d K;
    K.setIdentity();
    double fx = 1;//495;
    double fy = 1;//505;
    double cx = 0;//500;
    double cy = 0;//600;
    K(0, 0) = fx;
    K(1, 1) = fy;
    K(0, 2) = cx;
    K(1, 2) = cy;
    {

        for(int i=0; i<N; i++)
        {
            {
                Eigen::Vector3d Xc;
                Xc.setRandom();

                Eigen::Vector3d X = R_eig.transpose() * (Xc - t);

                Eigen::Vector3d uh = K * Xc;
                Eigen::Vector2d u = uh.segment<2>(0) / uh(2);

                // Monocular observation
                if(rand() % 2 == 0)
                {



                    Eigen::Vector2d v;
                    v.setRandom();

                    Eigen::Matrix<double,2,1> obs = u + v;

                    g2o::EdgeSE3CovProjectXYZOnlyPose* e = new g2o::EdgeSE3CovProjectXYZOnlyPose();
//                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = 1.0 / v.norm();


                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = fx;
                    e->fy = fy;
                    e->cx = cx;
                    e->cy = cy;
                    e->sigma_d_2 = 1.0;// / invSigma2;
                    if (covUseMode == 1) {

                        Eigen::Matrix3d S3d_eig;

                        S3d_eig.setRandom();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd(S3d_eig, Eigen::ComputeFullU | Eigen::ComputeFullV );
                        Eigen::Matrix3d S;
                        S.setZero();
                        for (int si = 0; si < 3; si++)
                        {
                            S(si, si) = fabs(svd.singularValues()[si]);
                        }
                        auto U = svd.matrixU();
                        S3d_eig = U * S * U.transpose();
//                        S3d_eig.setZero();
                        e->sigma_p_2 = (S3d_eig(0, 0) + S3d_eig(1, 1) + S3d_eig(2, 2)) * 0.33333 * invSigma2;
                        e->Sigma_p_2 = S3d_eig * invSigma2; //Eigen::Matrix3d::Identity() * e->sigma_p_2;//
                    } else {
                        e->sigma_p_2 = 0.0;
                        e->Sigma_p_2.setZero();
                    }
                    e->Xw[0] = X(0);
                    e->Xw[1] = X(1);
                    e->Xw[2] = X(2);
                    Eigen::Matrix2d Info = Eigen::Matrix2d::Identity() * invSigma2;
                    e->setInformation(Info);

                    //num diff
                    e->computeError();
                    Eigen::Vector2d y = e->error();
                    Eigen::Matrix<double, 2, 6> jac_num, jac_analyt;
                    double d = 1e-6;
                    for (int j = 0; j < 6; j++)
                    {
                        g2o::Vector6d upd;
                        upd.setZero();
                        upd(j) = d;
                        g2o::SE3Quat se3_upd = g2o::SE3Quat::exp(upd);
                        g2o::SE3Quat se3_m = se3_upd * se3_init;
                        vSE3->setEstimate(se3_m);
                        e->computeError();
                        Eigen::Vector2d y_p = e->error();
                        jac_num.col(j) = 1.0/d*(y_p-y);
                    }

                    vSE3->setEstimate(se3_init);

                    e->compute_jacobian(&jac_analyt);
                    double err = (jac_num-jac_analyt).norm();
                    double rel_err = err / (jac_num.norm() + 1e-3);
                    if (rel_err > 1e-3 || err > 1e-3)
                    {
                        std::cout << " mono: big error " << rel_err << " " << err << std::endl;
                        std::cout << " analyt " << std::endl;
                        std::cout << jac_analyt << std::endl;
                        std::cout << "num" << std::endl;
                        std::cout << jac_num << std::endl;
                    } else {
                        std::cout << " mono: OK error " << rel_err<< " " << err << std::endl;
                    }


//                    optimizer.addEdge(e);
//
//                    vpEdgesMono.push_back(e);
//                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation
                {
                    double b = 0.5;
                    Eigen::Vector3d urh = K * (Xc - Eigen::Vector3d(b, 0, 0));
                    Eigen::Vector2d ur = urh.segment<2>(0) / urh(2);


                    //SET EDGE
                    Eigen::Matrix<double,3,1> obs;

                    obs << u(0), u(1), ur(0);
                    Eigen::Vector3d v;
                    v.setRandom();
                    obs = obs + v;

                    g2o::EdgeStereoSE3CovProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3CovProjectXYZOnlyPose();
//                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = 1.0 / v.norm();


                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = fx;
                    e->fy = fy;
                    e->cx = cx;
                    e->cy = cy;
                    e->b = b;
                    e->sigma_d_2 = 1.0 ;
                    if (covUseMode == 1) {

                        Eigen::Matrix3d S3d_eig;

                        S3d_eig.setRandom();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd(S3d_eig, Eigen::ComputeFullU | Eigen::ComputeFullV );
                        Eigen::Matrix3d S;
                        S.setZero();
                        for (int si = 0; si < 3; si++)
                        {
                            S(si, si) = fabs(svd.singularValues()[si]);
                        }
                        auto U = svd.matrixU();
                        S3d_eig = U * S * U.transpose();
                        S3d_eig.setZero();
                        e->Sigma_p_2 = S3d_eig * invSigma2; //Eigen::Matrix3d::Identity() * e->sigma_p_2;//
                    } else {
                        e->sigma_p_2 = 0.0;
                        e->Sigma_p_2.setZero();
                    }
                    e->Xw[0] = X(0);
                    e->Xw[1] = X(1);
                    e->Xw[2] = X(2);
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    //num diff
                    e->computeError();
                    Eigen::Vector3d y = e->error();
                    Eigen::Matrix<double, 3, 6> jac_num, jac_analyt;
                    double d = 1e-5;
                    for (int j = 0; j < 6; j++)
                    {
                        g2o::Vector6d upd;
                        upd.setZero();
                        upd(j) = d;
                        g2o::SE3Quat se3_upd = g2o::SE3Quat::exp(upd);
                        g2o::SE3Quat se3_m = se3_upd * se3_init;
                        vSE3->setEstimate(se3_m);
                        e->computeError();
                        Eigen::Vector3d y_p = e->error();
                        jac_num.col(j) = 1.0/d*(y_p-y);
                    }

                    vSE3->setEstimate(se3_init);

                    e->compute_jacobian(&jac_analyt);
                    double err = (jac_num-jac_analyt).norm();
                    double rel_err = err / (jac_num.norm() + 1e-3);
                    if (rel_err > 1e-3 || err > 1e-3)
                    {
                        std::cout << " stereo: big error " << rel_err << " " << err << std::endl;
                        std::cout << " analyt " << std::endl;
                        std::cout << jac_analyt << std::endl;
                        std::cout << "num" << std::endl;
                        std::cout << jac_num << std::endl;
                    } else {
                        std::cout << " stereo: OK error " << rel_err<< " " << err << std::endl;
                    }

                }
            }
        }
    }

}


int main(int argc, char **argv)
{

//    test_refinement();
//    return 0;


    int n_modes = 2;
    bool isPoseOpt = true;


    if(argc < 10)
    {
        cerr << endl << "Usage: ./stereo_euroc path_to_vocabulary path_to_settings path_to_left_folder path_to_right_folder path_to_times_file" << endl;
        return 1;
    }

    std::cout << " starting slam " << std::endl;

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    LoadImages(string(argv[3]), string(argv[4]), string(argv[5]), vstrImageLeft, vstrImageRight, vTimeStamp);

    int trial = atoi(argv[6]);

    std::string pref = argv[7];

    int pnpMode = 0;
    if (argc > 8)
    {
        pnpMode = atoi(argv[8]);
    }

    int q = atoi(argv[9]);


    if(vstrImageLeft.empty() || vstrImageRight.empty())
    {
        cerr << "ERROR: No images in provided path." << endl;
        return 1;
    }

    if(vstrImageLeft.size()!=vstrImageRight.size())
    {
        cerr << "ERROR: Different number of left and right images." << endl;
        return 1;
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);


    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,false);

    std::cout << " using pnp mode " << pnpMode << std::endl;

    SLAM.ChangePnpMode(pnpMode, isPoseOpt, (pnpMode>0), false);

    sleep_ms(10e6);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    std::ofstream mom_traj_out(pref + "/MomCameraTrajectory_" + std::to_string(pnpMode) +"_" + std::to_string(trial) + ".txt");

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    for(int ni=0; ni<nImages; ni++)
    {
        if (ni % q != 0)
        {
            continue;
        }

        std::cout << ni << " " << ni % q << std::endl;

        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        if(imRight.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageRight[ni]) << endl;
            return 1;
        }

        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

        double tframe = vTimeStamp[ni];


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        cv::Mat Tcw = SLAM.TrackStereo(imLeftRect,imRightRect,tframe);
        if (Tcw.empty())
        {
            return 0;
        }

        for (int ii = 0; ii < 3; ii++)
        {
            for (int jj = 0; jj < 4; jj++)
            {
                mom_traj_out << Tcw.at<float>(ii, jj) << " ";
            }
        }
        mom_traj_out << std::endl;

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
            T = vTimeStamp[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimeStamp[ni-1];

        if(ttrack<T)
            sleep_ms((T-ttrack)*1e6);
    }

    SLAM.ActivateLocalizationMode();

    for (int i = 0; i < 10; i++)
        std::cout << "SLAM STOP" << std::endl;

    //cv::Mat Two = SLAM.SaveTrajectoryKITTI(pref + "/CameraTrajectory_" + std::to_string(trial) + ".txt");
    cv::Mat Two = SLAM.SaveTrajectoryKITTI(pref + "/CameraTrajectory_" + std::to_string(pnpMode) +"_" + std::to_string(trial) + ".txt");


    SLAM.Shutdown();
//
    return 0;




    sleep_ms(10e6);

    for (int i = 0; i < 10; i++)
        std::cout << " RELOC START " << nImages << std::endl;


    std::vector<std::string> mode_labels {"/epnp_", "/epnpu_", "dlsu_", "/dlsu_x_2_"};

    SLAM.ChangePnpMode(0, isPoseOpt, false, false);

    for (int reloc_trial = 0; reloc_trial < 3; reloc_trial++) {
        runLocalization(pref + "epnp_" + std::to_string(trial)+"_"+std::to_string(reloc_trial),
                        nImages, SLAM, Two, vTimeStamp, vstrImageLeft, vstrImageRight, M1l, M2l, M1r, M2r);
    }



    return 0;

    for (int mode = 1; mode < n_modes; mode++) {

        for (int thr_it = 2; thr_it < 3; thr_it ++)
        {
            double thr = 0.5 + thr_it * 0.25;
            SLAM.ChangePnpMode(mode, isPoseOpt, true, false, thr);
            runLocalization(pref + mode_labels[mode] + std::to_string(thr_it)+ "_" + std::to_string(trial),
                            nImages, SLAM, Two, vTimeStamp, vstrImageLeft, vstrImageRight, M1l, M2l, M1r, M2r);
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
    return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}
