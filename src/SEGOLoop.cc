/** This is an additional class for running SEGO inside a RANSAC loop
* Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
* Redistribution and use is allowed according to the terms of the GPL v3 license.
**/

#include "SEGOLoop.h"
#include "RANSACLoop.h"

#include <fstream>
#include "sego.h"
#include <opencv/cv.hpp>
#include "opencv2/core.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;

SEGOLoop::SEGOLoop(const vector<vector<vector<KeyPoint>>>& pt_triplet_colls, double p, double O,
                   const Matrix3d& K,
                   const vector<vector<Vector3d>>& pts3d_for_trips, double thr,
                   const vector<vector<int>>& tri_inds, const Vector3d& t12) :
                    RANSACLoop(6, p, O), K(K), t12(t12), pts3d_for_trips(pts3d_for_trips),
                    pt_triplet_colls(pt_triplet_colls), pix_thr(thr), tri_inds(tri_inds)
{
    //compute the number of triplet matches
    pt_num = 0;
    for (int ci = 0; ci < pt_triplet_colls.size(); ci++)
    {
        pt_num += pt_triplet_colls[ci].size();
    }

    trip_num = pt_triplet_colls.size();

    //put point triplets into matrices of projections (pt_num x 4, 2 channel double) and visibility (pt_num x 4, single-byte uchar)
    p_projs = Mat(pt_num, 4, CV_64FC2);
    vis_p = Mat::zeros(pt_num, 4, CV_8UC1);
    int ind = 0;
    for (int ci = 0; ci < pt_triplet_colls.size(); ci++)
    {
        for (int ti = 0; ti < pt_triplet_colls[ci].size(); ti++)
        {
            for (int pi = 0; pi < 3; pi++)
            {
                vis_p.at<uchar>(ind, tri_inds[ci][pi]) = 1;
                Vec2d& pp = p_projs.at<Vec2d>(ind, tri_inds[ci][pi]);
                Point2f pt = pt_triplet_colls[ci][ti][pi].pt;
                Vector3d xh;
                xh << pt.x,pt.y,1.0;
                xh = K.inverse() * xh;
                pp(0) = xh(0);
                pp(1) = xh(1);
            }
            all_inds.push_back(ind);
            ind++;
        }
    }

    is_inlier = vector<bool>(pt_num, false);
}

bool SEGOLoop::SolveOnce()
{
    bool solvable = false;
    vector<int> chinds;
    vector<int> curr_inds;

    //choose a point triplet of projections such that lines are not parallel
    Mat c_vis_p, c_p_projs;

    while (!solvable) {
        curr_inds = all_inds;
        chinds.clear();
        for (int i = 0; i < 3; i++) {
            int ri = rand() % (pt_num);
            chinds.push_back(curr_inds[ri]);
            curr_inds[ri] = curr_inds.back();
            curr_inds.pop_back();
        }

        int c_pt_num = 0;
        int c_ln_num = 0;
        for (int ci = 0; ci < chinds.size(); ci++) {
            if (chinds[ci] >= pt_num) {
                c_ln_num++;
            } else {
                c_pt_num++;
            }
        }

        c_vis_p = Mat::zeros (c_pt_num, 4, CV_8UC1);
        c_p_projs = Mat (c_pt_num, 4, CV_64FC2);

        int pti = 0;
        for (int ci = 0; ci < chinds.size(); ci++) {
            int pt_gi = chinds[ci];
            vis_p.row(pt_gi).copyTo(c_vis_p.row(pti));
            p_projs.row(pt_gi).copyTo(c_p_projs.row(pti));
            pti++;

        }

        solvable = true;
    }

    //run a SEGO or an Approx solver
    vector<Matrix3d> Rs;
    vector<Vector3d> ts;

    int64 t0 = getTickCount();


    bool is_right_left = false;
    Mat c_l_projs = Mat (0, 4, CV_64FC3);
    Mat c_vis_l = Mat::zeros (0, 4, CV_8UC1);
    sego_solver(c_p_projs, c_l_projs, c_vis_p, c_vis_l, true, is_right_left, &Rs, &ts);
    int64 t1 = getTickCount();

    //apply scale (solvers assume unit baseline)
    for (int i = 0; i < ts.size(); i++)
    {
        ts[i] = fabs(t12(0))*ts[i];
    }

    //find inliers
    for (int si = 0; si < Rs.size(); si++)
    {
        int curr_inliers = 0;
        vector<bool> is_cur_inlier(pt_num, false);
        int pfi = 0;
        int lfi = pt_num;

        vector<int> inliers_for_trips(trip_num, 0);
        for (int ti = 0; ti < trip_num; ti++)
        {
            Matrix3d Rc;
            Vector3d tc;
            if (ti == 0)
            {
                Rc = Rs[si];
                tc = ts[si];
            }
            if (ti == 1)
            {
                Rc = Rs[si];
                tc = ts[si]-Rc*t12+t12;
            }
            if (ti == 2)
            {
                Rc = Rs[si].transpose();
                tc = -Rc*ts[si];
            }
            if (ti == 3)
            {
                Rc = Rs[si].transpose();
                tc = -Rc*ts[si]-Rc*t12+t12;
            }

            for (int pi = 0; pi < pts3d_for_trips[ti].size(); pi++)
            {
                Vector3d Xc = K*(Rc*pts3d_for_trips[ti][pi]+tc);
                Xc = Xc/Xc(2);
                Vector3d xh;
                Point2f pt = pt_triplet_colls[ti][pi][2].pt;
                xh << pt.x, pt.y, 1.0;
                double pe = (Xc - xh).norm();
                if (pe < pix_thr)
                {
                    curr_inliers++;
                    is_cur_inlier[pfi] = true;
                    inliers_for_trips[ti]++;
                }
                pfi++;
            }
        }
        if (curr_inliers > inlier_cnt)
        {
            inlier_cnt = curr_inliers;
            is_inlier = is_cur_inlier;
            R_best = Rs[si];
            t_best = ts[si];
            double est_O = 1.0-inlier_cnt / (pt_num);
            if (est_O < O)
            {
                O = est_O;
            }
            best_inl_for_trips = inliers_for_trips;
        }
    }
    return true;
}

//visualize inliers
void SEGOLoop::DrawInliers(Mat img) {
    int gpi = 0;
    Scalar sego_color(0,0,255);
    for (int ti = 0; ti < tri_inds.size(); ti++) {
        for (int pi = 0; pi < pt_triplet_colls[ti].size(); pi++) {
            if (is_inlier[gpi]) {
                if (vis_p.at<uchar>(gpi, 0) == 1) {
                    Vec2d& pt_cv = p_projs.at<Vec2d>(gpi, 0);
                    Vector3d x;
                    x << pt_cv(0), pt_cv(1), 1.0;
                    x = K*x;
                    circle(img, Point(int(x(0)), int(x(1))), 3, sego_color, -1);
                }
            }
            gpi++;
        }
    }
}