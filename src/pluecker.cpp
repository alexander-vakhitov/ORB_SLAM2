//
// Created by alexander on 20.01.18.
//
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include <opencv/cxeigen.hpp>
#include "sego.h"

void analyze_case(const cv::Mat& vis, int ind, const cv::Mat& projs,
                  bool* is_direct, bool* is_stereoshift, Eigen::Vector3d* p1,
                  Eigen::Vector3d* p2, Eigen::Vector3d* third_proj)
{
//    std::cout << " check case " << std::endl;
    for (int i = 0; i < 4; i++)
    {
//        std::cout << (int)vis.at<uchar>(ind, i) << " " ;
    }
//    std::cout << std::endl;
    *is_direct = false;
    if ((int)vis.at<uchar>(ind, 0) + (int)vis.at<uchar>(ind, 1) == 2)
    {
        *is_direct = true;
    }
    *is_stereoshift = false;
    if ((*is_direct && (int)vis.at<uchar>(ind, 3) == 1) ||
            (!(*is_direct) && (int)vis.at<uchar>(ind, 1) == 1))
    {
        *is_stereoshift = true;
    }
    int c1, c2, c3;
    if (*is_direct) {
        c1 = 0;
        c2 = 1;
        if (*is_stereoshift) {
            c3 = 3;
        } else {
            c3 = 2;
        }
    } else {
        c1 = 2;
        c2 = 3;
        if (*is_stereoshift) {
            c3 = 1;
        } else {
            c3 = 0;
        }
    }

    if (projs.type() == CV_64FC2)
    {
        *p1 << projs.at<cv::Vec2d>(ind, c1)[0], projs.at<cv::Vec2d>(ind, c1)[1], 1.0;
        *p2 << projs.at<cv::Vec2d>(ind, c2)[0], projs.at<cv::Vec2d>(ind, c2)[1], 1.0;
        *third_proj << projs.at<cv::Vec2d>(ind, c3)[0], projs.at<cv::Vec2d>(ind, c3)[1], 1.0;
    } else {
        *p1 << projs.at<cv::Vec3d>(ind, c1)[0], projs.at<cv::Vec3d>(ind, c1)[1], projs.at<cv::Vec3d>(ind, c1)[2];
        *p2 << projs.at<cv::Vec3d>(ind, c2)[0], projs.at<cv::Vec3d>(ind, c2)[1], projs.at<cv::Vec3d>(ind, c2)[2];
        *third_proj << projs.at<cv::Vec3d>(ind, c3)[0], projs.at<cv::Vec3d>(ind, c3)[1], projs.at<cv::Vec3d>(ind, c3)[2];
    }
}

void get_urx_row_pt(const Eigen::Vector3d& p1, const Eigen::Matrix3i& inds, const Eigen::Vector3d& p3,
                    const Eigen::Vector3d& u, Eigen::Matrix<double, 1, 9>* c_row)
{
    Eigen::Matrix<double, 3, 9> C = Eigen::Matrix<double, 3, 9>::Zero();
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            C(j, inds(i,j)) = p1(i);
        }
    }
    Eigen::Matrix3d ux;
    GenerateCrossProductMatrix(u, &ux);
    *c_row = p3.transpose() * ux * C;
}

void get_stereoshift_row_pt(const Eigen::Vector3d& p1, const Eigen::Matrix3i& inds, const Eigen::Vector3d& t2,
                    const Eigen::Vector3d& p3, Eigen::Matrix<double, 1, 9>* c_row)
{
    Eigen::Matrix<double, 3, 9> C = Eigen::Matrix<double, 3, 9>::Zero();
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            C(j, inds(i,j)) = p1(i);
        }
    }
    Eigen::Matrix3d t2_x;
    GenerateCrossProductMatrix(t2, &t2_x);
    *c_row = p3.transpose() * t2_x * C;
}

void get_rps_row_pt(const Eigen::Vector3d& p1, const Eigen::Matrix3i& inds, const Eigen::Vector3d& p3,
                    const Eigen::Vector3d& pt_shift, Eigen::Matrix<double, 1, 9>* c_row)
{
    Eigen::Matrix3d pt_s_x;
    GenerateCrossProductMatrix(pt_shift, &pt_s_x);
    Eigen::Vector3d p_cross_1 = -pt_s_x * p1;
    Eigen::Matrix<double, 3, 9> C = Eigen::Matrix<double, 3, 9>::Zero();
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            C(j, inds(i,j)) = p_cross_1(i);
        }
    }
    *c_row = p3.transpose()  * C;
}

void generate_epipolar_eqs_pt(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3,
                              const Eigen::Vector3d& pt_shift, const Eigen::Vector3d& u, bool is_direct, bool is_stereoshift,
                              Eigen::Matrix<double, 2, 18>* A)
{
    Eigen::Matrix3i inds;
    inds << 0, 1, 2,
            3, 4, 5,
            6, 7, 8;
    Eigen::Vector3d t2;
    t2 << 1,0,0;
    if (!is_direct)
    {
        inds = Eigen::Matrix3i(inds.transpose());
    }
    Eigen::Vector3d p3_normed = p3/p3.norm();

    Eigen::Matrix<double, 1, 9> car1_row, car2_row, cr1_row, cr2_row;
    if (is_direct)
    {
        get_urx_row_pt(p1, inds, p3_normed, u, &car1_row);
        get_urx_row_pt(p2, inds, p3_normed, u, &car2_row);
        get_rps_row_pt(p1, inds, p3_normed, pt_shift, &cr1_row);
        get_rps_row_pt(p2, inds, p3_normed, pt_shift+t2, &cr2_row);
    } else {
        get_rps_row_pt(p1, inds, p3_normed, u, &car1_row);
        get_rps_row_pt(p2, inds, p3_normed, u, &car2_row);
        get_urx_row_pt(p1, inds, p3_normed, pt_shift, &cr1_row);
        get_urx_row_pt(p2, inds, p3_normed, pt_shift, &cr2_row);
        Eigen::Matrix<double, 1, 9> cr2_add_row;
        get_rps_row_pt(p2, inds, p3, t2, &cr2_add_row);
        cr2_row = cr2_row + cr2_add_row;
    }
    if (is_stereoshift)
    {
        Eigen::Matrix<double, 1, 9> css1_row;
        get_stereoshift_row_pt(p1, inds, t2, p3_normed, &css1_row);
        Eigen::Matrix<double, 1, 9> css2_row;
        get_stereoshift_row_pt(p2, inds, t2, p3_normed, &css2_row);
        cr1_row = cr1_row + css1_row;
        cr2_row = cr2_row + css2_row;
    }
    A->block<1,9>(0,0) = cr1_row;
    A->block<1,9>(0,9) = car1_row;
    A->block<1,9>(1,0) = cr2_row;
    A->block<1,9>(1,9) = car2_row;
}

void get_rx_mat(const Eigen::Vector3d& Xm, const Eigen::Vector3d& p3, const Eigen::Matrix3i& inds,
                Eigen::Matrix<double, 3, 9>* A)
{
//    std::cout << " Xm: " << Xm << std::endl;
//    std::cout << " p3: " << p3 << std::endl;
//    std::cout << " inds row 1: " << inds(0,0) << " " << inds(0, 1) << " " << inds(0, 2) << std::endl;
    Eigen::Matrix<double, 3, 9> C;
    C.setZero();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            C(i, inds(j, i)) = Xm(j);
        }
    }
    Eigen::Matrix3d p3_x;
    GenerateCrossProductMatrix(p3, &p3_x);
    *A = p3_x * C;

//    std::cout << " line mat " << std::endl;
//    for (int i = 0; i < 3; i++)
//    {
//        for (int j = 0; j < 9; j++)
//        {
//            std::cout << (*A)(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
}

void get_urx_row_ln(const Eigen::Vector3d& p1, const Eigen::Matrix3i& inds, const Eigen::Matrix3d& p3x,
                    const Eigen::Vector3d& u, Eigen::Matrix<double, 3, 9>* c_row)
{
    Eigen::Matrix<double, 3, 9> C = Eigen::Matrix<double, 3, 9>::Zero();
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            C(j, inds(i,j)) = p1(i);
        }
    }
    Eigen::Matrix3d ux;
    GenerateCrossProductMatrix(u, &ux);
    *c_row = p3x.transpose() * ux * C;
//    std::cout << "p3x" << std::endl;
//    for (int i = 0; i < 3; i++)
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            std::cout << p3x(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << "ux" << std::endl;
//    for (int i = 0; i < 3; i++)
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            std::cout << ux(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << " line row " << std::endl;
//    for (int j = 0; j < 3; j++) {
//        for (int i = 0; i < 9; i++) {
//            std::cout << (*c_row)(j, i) << " ";
//        }
//    }
//    std::cout << std::endl;
}


bool generate_epipolar_eqs_ln(const Eigen::Vector3d& l1, const Eigen::Vector3d& l2, const Eigen::Vector3d& l3,
                              const Eigen::Vector3d pt_shift, const Eigen::Vector3d& u, bool is_direct, bool is_stereoshift,
                              Eigen::Matrix<double, 2, 18>* A)
{
    Eigen::Matrix3i inds;
    inds << 0, 1, 2,
            3, 4, 5,
            6, 7, 8;
    Eigen::Vector3d t2;
    t2 << 1,0,0;
    if (!is_direct)
    {
//        std::cout << " transposing " << inds(0,0) << " " << inds(0,1) << std::endl;

        inds = Eigen::Matrix3i(inds.transpose());
//        std::cout << " after transposing " << inds(0,0) << " " << inds(0,1) << std::endl;
    }
    Eigen::Vector3d X1, X2;
    if (!TriangulateLine(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), t2, l1, l2, &X1, &X2))
    {
        return false;
    }
    Eigen::Vector3d Xcp = X1.cross(X2);
    Eigen::Vector3d dX = X2-X1;
    Eigen::Matrix<double, 3, 9> r_eqs;
    get_rx_mat(Xcp, l3, inds, &r_eqs);
    Eigen::Matrix3d l3xt;
    GenerateCrossProductMatrix(l3, &l3xt);
    l3xt = Eigen::Matrix3d(l3xt.transpose());
    Eigen::Matrix<double, 3, 9> ar_eqs;
    Eigen::Matrix<double, 3, 9> r_eqs_add;
    if (is_direct)
    {
        get_urx_row_ln(dX, inds, l3xt, u, &ar_eqs);
        Eigen::Vector3d Xm = -pt_shift.cross(dX);
        get_rx_mat(Xm, l3, inds, &r_eqs_add);
    } else {
        Eigen::Vector3d nDx = -u.cross(dX);
        get_rx_mat(nDx, l3, inds, &ar_eqs);
        get_urx_row_ln(dX, inds, l3xt, pt_shift, &r_eqs_add);
    };
    if (is_stereoshift)
    {
        Eigen::Matrix<double, 3, 9> r_eqs_stereo_add;
        get_urx_row_ln(dX, inds, l3xt, t2, &r_eqs_stereo_add);
        r_eqs = r_eqs + r_eqs_stereo_add;
    }
    Eigen::Matrix<double, 3, 18> A_tmp;
    A_tmp.block<3,9>(0,0) = r_eqs + r_eqs_add;
    A_tmp.block<3,9>(0,9) = ar_eqs;
//    std::cout << " r_eqs " << std::endl;
//    for (int i = 0; i < 3; i++)
//    {
//        for (int j = 0; j < 9; j++)
//        {
//            std::cout << r_eqs(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << " r_eqs_add " << std::endl;
//    for (int i = 0; i < 3; i++)
//    {
//        for (int j = 0; j < 9; j++)
//        {
//            std::cout << r_eqs_add(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
    *A = Eigen::Matrix<double,2,18>(A_tmp.block<2,18>(0,0));
    return true;
}


bool eq_gen_plu_real(const cv::Mat& projs, const cv::Mat& lprojs, const cv::Mat& vis_p, const cv::Mat& vis_l,
                     Eigen::Matrix<double, 4, 18>* A, Eigen::Vector3d* pt_shift)
{
    Eigen::Vector3d t2;
    t2 << 1,0,0;
    cv::Mat t2_cv;
    cv::eigen2cv(t2, t2_cv);

    cv::Point2d pt1(projs.at<cv::Vec2d>(0, 0)[0], projs.at<cv::Vec2d>(0, 0)[1]);
    cv::Point2d pt2(projs.at<cv::Vec2d>(0, 1)[0], projs.at<cv::Vec2d>(0, 1)[1]);
//        std::vector<cv::Point3d> pts_3d;
    cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64FC1);
    cv::Mat eye_mat = cv::Mat::eye(3, 3, CV_64FC1);
    eye_mat.copyTo(P0(cv::Rect(0, 0, 3, 3)));
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64FC1);
    eye_mat.copyTo(P1(cv::Rect(0, 0, 3, 3)));
    t2_cv.copyTo(P1(cv::Rect(3, 0, 1, 3)));
    cv::Mat pts_3d;
    cv::triangulatePoints(P0, P1, std::vector<cv::Point2d>{pt1}, std::vector<cv::Point2d>{pt2}, pts_3d);
    *pt_shift <<  pts_3d.at<double>(0, 0), pts_3d.at<double>(1, 0), pts_3d.at<double>(2,0);
    *pt_shift = *pt_shift / pts_3d.at<double>(3, 0);

//    std::ofstream log_out ("/home/alexander/debug_pluecker_gen");
//    log_out << (*pt_shift)(0) << " " << (*pt_shift)(1) << " " << (*pt_shift)(2) << std::endl;

    Eigen::Vector3d u ;
    u << projs.at<cv::Vec2d>(0, 2)[0],projs.at<cv::Vec2d>(0, 2)[1],1.0;

    int a_ind = 0;

    for (int pi = 1; pi < vis_p.rows; pi++)
    {
        bool is_direct, is_stereoshift;
        Eigen::Vector3d p1, p2, p3;
        analyze_case(vis_p, pi, projs, &is_direct, &is_stereoshift, &p1, &p2, &p3);
//        std::cout << " pt " << pi << " " << is_direct << " " << is_stereoshift << std::endl;
        Eigen::Matrix<double, 2, 18> A_curr;
        generate_epipolar_eqs_pt(p1, p2, p3, *pt_shift, u, is_direct, is_stereoshift, &A_curr);
        A->block<2,18>(2*a_ind, 0) = A_curr;
        a_ind += 1;
    }

    for (int li = 0; li < vis_l.rows; li++)
    {
        bool is_direct, is_stereoshift;
        Eigen::Vector3d l1, l2, l3;
        analyze_case(vis_l, li, lprojs, &is_direct, &is_stereoshift, &l1, &l2, &l3);
//        std::cout << " ln " << li << " " << is_direct << " " << is_stereoshift << std::endl;
        Eigen::Matrix<double, 2, 18> A_ln;
        if (!generate_epipolar_eqs_ln(l1, l2, l3, *pt_shift, u, is_direct, is_stereoshift, &A_ln))
        {
            return false;
        }
        A->block<2,18>(2*a_ind, 0) = A_ln;
        a_ind += 1;
    }
    return true;
}