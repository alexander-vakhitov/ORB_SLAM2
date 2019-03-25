//
// Created by alexander on 14.03.18.
//
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "sego.h"

#include "approx_relpose_generalized_fast.h"

void fill_c_row(const Eigen::Matrix<double,6,1>& u, const Eigen::Matrix<double,6,1>& v, const Eigen::Matrix3d& R, Eigen::Matrix<double,1,4>& c_row)
{
    Eigen::Vector3d um = R*u.block<3,1>(0,0);
    Eigen::Matrix3d M;
    GenerateCrossProductMatrix(um, &M);
    c_row.block<1,3>(0,0) = v.block<3,1>(0,0).transpose() * M;
    Eigen::Vector3d vr = R*u.block<3,1>(3,0);
    Eigen::Vector3d ur = R*u.block<3,1>(0,0);
    double a = vr.dot(v.block<3,1>(0,0)) + ur.dot(v.block<3,1>(3,0));
    c_row(0,3) = a;
}

bool approx(const cv::Mat& projs, const cv::Mat& vis_p,
                 std::vector<Eigen::Matrix3d>* Rs, std::vector<Eigen::Vector3d>* ts)
{
//    std::cout << " started "<< std::endl;
    Eigen::Vector3d b;
    b << -1,0,0;
//    double* wvec = new double[36*6];

    Eigen::Matrix<double, 6, 6> us, vs;
    std::vector<Eigen::Matrix<double,6,6>> wv;

    int ind = 0;
    int ind_w = 0;
    for (int pi = 0; pi < 3; pi++)
    {
        for (int mi = 0; mi < 2; mi++)
        {
            for (int oi = 2; oi < 4; oi++)
            {
                if (vis_p.at<uchar>(pi, mi) == 1 && vis_p.at<uchar>(pi, oi) == 1)
                {
                    Eigen::Vector3d n;
                    n.setZero();

                    Eigen::Vector3d x_mi;
                    const cv::Vec2d& x_mi_cv = projs.at<cv::Vec2d>(pi, mi);
                    x_mi << x_mi_cv(0), x_mi_cv(1), 1.0;
                    if (mi == 1)
                    {
                        n = b.cross(x_mi);
                    }
                    Eigen::Matrix<double,6,1> u;
                    u.block<3,1>(0,0) = x_mi;
                    u.block<3,1>(3,0) = n;

                    Eigen::Vector3d n2;
                    n2.setZero();
                    const cv::Vec2d& x_oi_cv = projs.at<cv::Vec2d>(pi, oi);
                    Eigen::Vector3d x_oi;
                    x_oi << x_oi_cv(0), x_oi_cv(1), 1.0;
                    if (oi == 3)
                    {
                        n2 = b.cross(x_oi);
                    }
                    Eigen::Matrix<double,6,1> v;
                    v.block<3,1>(0,0) = x_oi;
                    v.block<3,1>(3,0) = n2;
                    Eigen::Matrix<double,6,6> w = u * v.transpose();
                    us.col(ind) = u;
                    vs.col(ind) = v;
                    ind++;
                    wv.push_back(w);
                }
            }
        }
    }

//    std::cout << " formed "<< std::endl;

    std::vector<Eigen::Vector3d> rsolns;
    approx_relpose_generalized_fast(wv[0], wv[1], wv[2], wv[3], wv[4], wv[5], rsolns);

//    std::cout << " solved in fact "<< std::endl;

    for (int i = 0; (i < rsolns.size()) && (i < 20); i++)
    {
        Eigen::Matrix3d CP;
        GenerateCrossProductMatrix(rsolns[i], &CP);
        Eigen::Matrix3d R = CP + Eigen::Matrix3d::Identity();
        Eigen::Matrix<double,6,4> C;
        for (int j = 0; j < 6; j++)
        {
            Eigen::Matrix<double,1,4> c_row;
            fill_c_row(us.col(j), vs.col(j), R, c_row);
            C.row(j) = c_row;
        }
        Eigen::Matrix<double,4,4> CtC = C.transpose() * C;
        Eigen::Matrix<double,4,1> ev = CtC.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV().col(3);
        Eigen::Vector3d t = - ev.block<3,1>(0,0)/ev(3,0);
        ts->push_back(t);
        Rs->push_back(R);
    }

    return true;
//    delete[] wvec;
}