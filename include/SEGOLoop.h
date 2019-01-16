/** This is an additional class for running SEGO inside a RANSAC loop
* Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
* Redistribution and use is allowed according to the terms of the GPL v3 license.
**/

#ifndef LINEDESCDATACOLLECTOR_SEGOLOOP_H
#define LINEDESCDATACOLLECTOR_SEGOLOOP_H
#include <Eigen/Dense>
#include <opencv2/core/types.hpp>
#include "RANSACLoop.h"

class SEGOLoop : public RANSACLoop {
public:
//Construct a SEGO/Approx RANSAC loop
//in:
//pt_triplet_colls - vector of 4 collections of dim n of vectors of dim 3 of matched keypoints,
// every collection is for a specific combination of views defined in tri_inds
//p - required success probability
//O - estimated inlier share
//K - intrinsics
//pts3d_for_trips - collection of vectors of triangulated 3d points for every combination of views
//thr - reprojection threshold in pixels
//tri_inds - vector of vectors defining view combinations
//ln_triplet_colls - vector of 4 collections of dim n of vectors of dim 3 of matched keylines,
// every collection is for a specific combination of views defined in tri_inds
//t12 - translation vector from 1st to 2nd view, t12=(x,0,0)
//isSEGO - flag defining whether SEGO or Approx algorithm should be used
    SEGOLoop(const std::vector<std::vector<std::vector<cv::KeyPoint>>>& pt_triplet_colls, double p, double O,
             const Eigen::Matrix3d& K,
             const std::vector<std::vector<Eigen::Vector3d>>& pts3d_for_trips, double thr,
             const std::vector<std::vector<int>>& tri_inds, const Eigen::Vector3d& t12);

//one RANSAC iteration
    bool SolveOnce() override;
//draw inliers
    void DrawInliers(cv::Mat img) override;

    std::vector<std::vector<Eigen::Vector3d>> pts3d_for_trips;
    std::vector<std::vector<std::vector<cv::KeyPoint>>> pt_triplet_colls;
    std::vector<std::vector<Eigen::Vector3d>> all_leqs3;
    Eigen::Matrix3d K;
    Eigen::Vector3d t12;

    int pt_num;
    double pix_thr;// = 5;
    std::vector<int> all_inds;
    std::vector<std::vector<int>> tri_inds;
    cv::Mat p_projs, vis_p;
    cv::Mat l_projs, vis_l;

    int trip_num;

    std::vector<int> best_inl_for_trips;
};


#endif //LINEDESCDATACOLLECTOR_SEGOLOOP_H
