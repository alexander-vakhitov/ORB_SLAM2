/** This is a parent class for a RANSAC loop
* Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
* Redistribution and use is allowed according to the terms of the GPL v3 license.
**/

#ifndef LINEDESCDATACOLLECTOR_RANSACLOOP_H
#define LINEDESCDATACOLLECTOR_RANSACLOOP_H
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/mat.hpp>

class RANSACLoop {
public:
    //construct RANSAC loop base object
    //in:
    //s - number of matches
    //p - required success probability
    //O - estimated inlier share
    RANSACLoop(double s, double p, double O): s(s), p(p), O(O), inlier_cnt(0)
    {
        RecomputeNIt();
    };

    virtual ~RANSACLoop() {};

    void Iterate(cv::Mat img = cv::Mat());

    virtual void DrawInliers(cv::Mat img) {};

    virtual bool SolveOnce() = 0;

    void RecomputeNIt()
    {
        double I = 1;
        for (int i = 0; i < s; i++)
        {
            I *= (1-O);
        }
//        std::cout << O << " " << " " << inlier_cnt << std::endl;
        Nit = log(1-p) / log(1-I);
    }



    Eigen::Matrix3d R_best;
    Eigen::Vector3d t_best;
    int inlier_cnt;
    int Nit;
    double s;
    double p;
    double O;

    std::vector<bool> is_inlier;
};


#endif //LINEDESCDATACOLLECTOR_RANSACLOOP_H
