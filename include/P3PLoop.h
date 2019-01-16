/** This is an additional class for running P3P inside a RANSAC loop
* Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
* Redistribution and use is allowed according to the terms of the GPL v3 license.
**/

#ifndef LINEDESCDATACOLLECTOR_P3PLOOP_H
#define LINEDESCDATACOLLECTOR_P3PLOOP_H
#include "RANSACLoop.h"

class P3PLoop : public RANSACLoop
{

public:
//Construct a P3P RANSAC loop
//in:
//pt_triplets - vecctor of dim n of vectors of dim 3 of matched keypoints
//p - required success probability
//O - estimated inlier share
//K - intrinsics
//pts3d - triangulated 3d points
//thr - reprojection threshold in pixels
    P3PLoop(const std::vector<std::vector<cv::KeyPoint>>& pt_triplets, double p, double O, const Eigen::Matrix3d& K,
            const std::vector<Eigen::Vector3d>& pts3d, double thr);

//make one iteration
    bool SolveOnce() override;

    std::vector<Eigen::Vector3d> pts3d;
    std::vector<std::vector<cv::KeyPoint>> pt_triplets_fin;
    Eigen::Matrix3d K;
    Eigen::Vector3d t12;

    double pix_thr;
    std::vector<int> all_inds;
};


#endif //LINEDESCDATACOLLECTOR_P3PLOOP_H
