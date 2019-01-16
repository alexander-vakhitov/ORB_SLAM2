/** This is an additional class for running P3P inside a RANSAC loop
* Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
* Redistribution and use is allowed according to the terms of the GPL v3 license.
**/
#include <Eigen/Dense>
#include <opencv2/core/types.hpp>
#include <opencv/cv.hpp>
#include <opencv/cxeigen.hpp>
#include <iostream>
#include "P3PLoop.h"
#include "P3P.h"

//constructor: initialize an is-inlier flag array and an array of match indices used in a RANSAC loop
P3PLoop::P3PLoop(const std::vector<std::vector<cv::KeyPoint>>& pt_triplets, double p, double O,
                 const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& pts3d, double thr) :
        RANSACLoop(6, p, O), K(K), t12(t12), pts3d(pts3d), pt_triplets_fin(pt_triplets), pix_thr(thr)
{
    for (int i = 0; i < pt_triplets.size(); i++)
    {
        all_inds.push_back(i);
    }
    is_inlier = std::vector<bool>(pt_triplets.size(), false);
}

bool P3PLoop::SolveOnce()
{
    std::vector<int> curr_inds = all_inds;

    //choose three point matches t0, p1, p2
    std::vector<int> chosen_inds;
    for (int i = 0; i < 3; i++)
    {
        int ri = rand() % curr_inds.size();
        chosen_inds.push_back(curr_inds[ri]);
        curr_inds[ri] = curr_inds.back();
        curr_inds.pop_back();
    }

    // we need to rotate the 3D points so that they lie in a z=0 plane
    // fill matrices of vectors p1-t0, p2-t0
    cv::Mat templ_sample = cv::Mat::zeros(2, 3, CV_64FC1);
    cv::Mat det_sample = cv::Mat::ones(2, 3, CV_64FC1);
    cv::Mat t0 (3,1,CV_64FC1);
    cv::Mat p1 (3,1,CV_64FC1);
    cv::Mat p2 (3,1,CV_64FC1);
    for (int i = 0; i < 3; i++)
    {
        t0.at<double>(i, 0) = pts3d[chosen_inds[0]](i);
        p1.at<double>(i, 0) = pts3d[chosen_inds[1]](i);
        p2.at<double>(i, 0) = pts3d[chosen_inds[2]](i);
    }

    cv::Mat r1 = p1-t0;
    r1 = r1 / cv::norm(r1);
    cv::Mat r3 = r1.cross(p2-t0);
    r3 = r3/cv::norm(r3);
    cv::Mat r2 = r3.cross(r1);

    cv::Mat R0(3,3,CV_64FC1);

    r1.copyTo(R0.col(0));
    r2.copyTo(R0.col(1));
    r3.copyTo(R0.col(2));

    R0 = R0.t();

    t0 = R0*t0;

    templ_sample.at<double>(0, 1) = r1.dot(p1);
    templ_sample.at<double>(1, 1) = r2.dot(p1);
    templ_sample.at<double>(0, 2) = r1.dot(p2);
    templ_sample.at<double>(1, 2) = r2.dot(p2);

    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3d xh;
        cv::Point2f pt = pt_triplets_fin[chosen_inds[i]][2].pt;
        xh << pt.x, pt.y, 1.0;
        xh = K.inverse() * xh;
        for (int j = 0; j < 2; j++)
        {
            det_sample.at<double>(j, i) = xh(j);
        }
    }
    std::vector<cv::Mat> Rs, ts;

    auto t00 = cv::getTickCount();

    //execute the P3P algorithm
    p3p(det_sample, templ_sample, Rs, ts);

    auto t1 = cv::getTickCount();

    for (int j = 0; j < Rs.size(); j++)
    {
        cv::Mat Rf = Rs[j]*R0 ;
        cv::Mat tf = Rs[j]*t0 + ts[j];
        Eigen::Matrix3d Rfe;
        Eigen::Vector3d tfe;
        cv::cv2eigen(Rf, Rfe);
        cv::cv2eigen(tf, tfe);
        int cur_inlier = 0;
        std::vector<bool> is_cur_inlier(pts3d.size(), false);
        //check for inliers
        for (int k = 0; k < pts3d.size(); k++)
        {
            Eigen::Vector3d Xc = K*(Rfe *pts3d[k] + tfe);
            Xc = Xc/Xc(2);
            Eigen::Vector3d Xc1_pred;
            Xc1_pred << pt_triplets_fin[k][2].pt.x, pt_triplets_fin[k][2].pt.y, 1.0;
            double pe = (Xc-Xc1_pred).norm();
            if (pe < pix_thr)
            {
                cur_inlier++;
                is_cur_inlier[k] = true;
            }
        }
        //update the best inlier number and the outlier share estimate
        if (cur_inlier > inlier_cnt)
        {
            inlier_cnt = cur_inlier;
            R_best = Rfe;
            t_best = tfe;
            double est_O = 1.0-inlier_cnt / (pts3d.size()+0.0);
            if (est_O < O)
            {
                O = est_O;
            }
            is_inlier = is_cur_inlier;
        }
    }
    return true;
}
