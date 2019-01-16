/** This is an implementation of a P3P algorithm from a paper
* Gao, Xiao-Shan and Hou, Xiao-Rong and Tang, Jianliang and Cheng, Hang-Fei.
* Complete solution classification for the perspective-three-point problem //
* IEEE TPAMI, 2003
*
* Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
* Redistribution and use is allowed according to the terms of the GPL v3 license.
**/

#include "P3P.h"

#include <opencv2/core.hpp>

#include <gsl/gsl_poly.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <iostream>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <gsl/gsl_eigen.h>

//
//void select_points(cv::Mat& m, double* pt_array, int pt_num, int pt_ind, int sel_ind, int coord_num)
//{
//    for (int i = 0; i < coord_num; i++)
//        m.at<double>(i, sel_ind) = pt_array[i * pt_num + pt_ind];
//}
//
//
//void normalize_projections(cv::Mat& det_pts, cv::Mat& det_norm_pts, cv::Mat& KC)
//{
//    det_norm_pts = cv::Mat::zeros(det_pts.rows, det_pts.cols, CV_64FC1);
//    for (int i = 0; i < det_pts.cols; i++)
//    {
//        det_norm_pts.at<double>(0, i) = (det_pts.at<double>(0, i) - KC.at<double>(0, 2))/KC.at<double>(0,0);
//        det_norm_pts.at<double>(1, i) = (det_pts.at<double>(1, i) - KC.at<double>(1, 2))/KC.at<double>(1,1);
//    }
//}
//
//
//void ransac_p3p_planar(double* templ_pts, int templ_pt_num,
//                double* det_pts, double* KC_arr, double ransac_thr, double* bestp)
//{
//    if (templ_pt_num < 3)
//    {
//        for (int i = 0; i < 6; i++) {
//            bestp[i] = 0.0;
//        }
//        return;
//    }
//
//    int MaxIt = 1000;
//    int it = 0;
//    int ind1=-1, ind2=-1, ind3=-1;
//
//    cv::Mat KC = cv::Mat::eye(3,3,CV_64FC1);
//    KC.at<double>(0,0) = KC_arr[0];
//    KC.at<double>(0,2) = KC_arr[2];
//    KC.at<double>(1,1) = KC_arr[4];
//    KC.at<double>(1,2) = KC_arr[5];
//
//
//
////    std::ofstream log_ransac("/home/alexander/ransac_log");
////    for (int i = 0; i < 3; i ++)
////    {
////        for (int j = 0; j < 3; j++)
////        {
////            log_ransac << KC.at<double>(i, j) << " ";
////        }
////        log_ransac << std::endl;
////    }
////    log_ransac.flush();
//
//    cv::Mat templ_pts_mat(2, templ_pt_num, CV_64FC1, templ_pts);
//    cv::Mat templ_pts_h = cv::Mat::zeros(3, templ_pt_num, CV_64FC1);
//    templ_pts_mat.copyTo(templ_pts_h(cv::Rect(0, 0, templ_pt_num, 2)));
//    cv::Mat det_pts_mat(2, templ_pt_num, CV_64FC1, det_pts);
//
////    for (int i = 0; i < templ_pt_num; i++)
////    {
////        log_ransac << det_pts_mat.at<double>(0, i) << " " << det_pts_mat.at<double>(1, i) << std::endl;
////    }
////    log_ransac.flush();
////
////    log_ransac << " template " << std::endl;
////    for (int i = 0; i < templ_pt_num; i++)
////    {
////        log_ransac << templ_pts_mat.at<double>(0, i) << " " << templ_pts_mat.at<double>(1, i) << std::endl;
////    }
////    log_ransac.flush();
//
//    int last_good_num = 3;
//    cv::Mat best_p_mat;
//    std::vector<int> best_inds;
//    double confidence = 0.99;
//
//    while (it < MaxIt)
//    {
//        //choose randomly 3 pts
//        ind1 = -1;
//        ind2 = -1;
//        ind3 = -1;
//        ind1 = rand() % templ_pt_num;
//        while (ind2 < 0 || ind2 == ind1)
//            ind2 = rand() % templ_pt_num;
//        while (ind3 < 0 || ind3 == ind1 || ind3 == ind2)
//            ind3 = rand() % templ_pt_num;
//
//        cv::Mat templ_sample = cv::Mat::zeros(2, 3, CV_64FC1);
//        cv::Mat det_sample = cv::Mat::ones(2, 3, CV_64FC1);
//        select_points(templ_sample, templ_pts, templ_pt_num, ind1, 0, 2);
//        select_points(templ_sample, templ_pts, templ_pt_num, ind2, 1, 2);
//        select_points(templ_sample, templ_pts, templ_pt_num, ind3, 2, 2);
//
//        select_points(det_sample, det_pts, templ_pt_num, ind1, 0, 2);
//        select_points(det_sample, det_pts, templ_pt_num, ind2, 1, 2);
//        select_points(det_sample, det_pts, templ_pt_num, ind3, 2, 2);
//
//        cv::Mat det_sample_norm;
//        normalize_projections(det_sample, det_sample_norm, KC);
//
//        cv::Mat templ_sample_h = cv::Mat::ones(3, 3, CV_64FC1);
//        templ_sample.copyTo(templ_sample_h(cv::Rect(0, 0, 3, 2)));
//
//        cv::Mat v1 = templ_sample_h.col(1) - templ_sample_h.col(0);
//        cv::Mat v2 = templ_sample_h.col(2) - templ_sample_h.col(0);
//        cv::Mat v1n = v1 / cv::norm(v1);
//        cv::Mat v2n = v2/cv::norm(v2);
//        if (cv::norm(v1.cross(v2)) < 1e-10 || fabs(v1n.dot(v2n))>0.9)
//        {
//            it += 1;
//            continue;
//        }
//        std::vector<cv::Mat> Rs, ts;
//
////        log_ransac << ind1 << " " << ind2 << " " << ind3 << std::endl;
////        for (int pi = 0; pi < 3; pi++)
////        {
////            log_ransac << det_sample.at<double>(0, pi) << " " << det_sample.at<double>(1, pi) << std::endl;
////            log_ransac << templ_sample.at<double>(0, pi) << " " << templ_sample.at<double>(1, pi) << std::endl;
////        }
////        log_ransac.flush();
//
//        p3p(det_sample_norm, templ_sample, Rs, ts);
//
//        for (int si = 0; si < Rs.size(); si++)
//        {
//            cv::Mat t_arr = cv::repeat(ts[si], 1, templ_pt_num);
//            cv::Mat rot_pts = Rs[si] * templ_pts_h;
//            cv::Mat Xc = KC*(rot_pts + t_arr);
//            cv::Mat Xc3 = cv::repeat(Xc(cv::Rect(0, 2, templ_pt_num, 1)), 2, 1);
//            cv::Mat Xc2;
////            std::cout << " xc3 " << std::endl;
////            std::cout << Xc3.col(0) << std::endl;
//            cv::divide(Xc(cv::Rect(0,0,templ_pt_num, 2)), Xc3, Xc2);
////            std::cout << " xc " << std::endl;
////            std::cout << Xc.col(0) << std::endl;
////            std::cout << " xc2 " << std::endl;
////            std::cout << Xc2.col(0) << std::endl;
////            std::cout << " diff " << std::endl;
//            cv::Mat diff = Xc2 - det_pts_mat;
////            std::cout << diff.col(0) << std::endl;
//            cv::Mat d2;
//            cv::multiply(diff, diff, d2);
//            cv::Mat errs_sq;
//            cv::reduce(d2, errs_sq, 0, CV_REDUCE_SUM);
//            cv::Mat errs;
//            cv::sqrt(errs_sq, errs);
//
////            std::cout << errs.col(0) << std::endl;
//
//            std::vector<int> good_inds;
//            for (int pi = 0; pi < templ_pt_num; pi++)
//            {
//                if (errs.at<double>(0, pi) < ransac_thr)
//                {
////                    std::cout << errs.at<double>(0, pi) << std::endl;
//                    good_inds.push_back(pi);
//                }
//            }
//            if (good_inds.size() > last_good_num)
//            {
//                cv::Mat p = cv::Mat(6, 1, CV_64FC1);
//                cv::Mat p_rot;
//                cv::Rodrigues(Rs[si], p_rot);
//                p_rot.copyTo(p(cv::Rect(0, 0, 1, 3)));
//                cv::Mat Rt;
//                cv::transpose(Rs[si], Rt);
//                cv::Mat c = -Rt*ts[si];
//                c.copyTo(p(cv::Rect(0, 3, 1, 3)));
//                best_p_mat = p;
//                best_inds = good_inds;
//                last_good_num = good_inds.size();
//                double ep = (templ_pt_num - last_good_num + 0.0) / templ_pt_num;
//                MaxIt = (int)floor(log(1-confidence) / log(1-pow(1-ep, 3)))+1;
//            }
//        }
//
//        it += 1;
//    }
//
//    if (last_good_num > 3) {
//        for (int i = 0; i < 6; i++) {
//            bestp[i] = best_p_mat.at<double>(i, 0);
//        }
//    } else {
//        for (int i = 0; i < 6; i++) {
//            bestp[i] = 0.0;
//        }
//    }
//}

void solve_deg4_companion(double a, double b, double c, double d, double e,
                     std::vector<double>& x)
{


    double data[] = { 0, 0, 0, -e/a,
                      1,0,0,-d/a,
                      0,1,0,-c/a,
                      0,0,1,-b/a};


    gsl_matrix_view m
            = gsl_matrix_view_array (data, 4, 4);

    gsl_vector_complex *eval = gsl_vector_complex_alloc (4);
    gsl_matrix_complex *evec = gsl_matrix_complex_alloc (4, 4);

    gsl_eigen_nonsymmv_workspace * w =
            gsl_eigen_nonsymmv_alloc (4);

    gsl_eigen_nonsymmv (&m.matrix, eval, evec, w);

    gsl_eigen_nonsymmv_free (w);

    gsl_eigen_nonsymmv_sort (eval, evec,
                             GSL_EIGEN_SORT_ABS_DESC);

    x.clear();

    int i, j;

    for (i = 0; i < 4; i++)
    {
        gsl_complex eval_i
                = gsl_vector_complex_get (eval, i);
        gsl_vector_complex_view evec_i
                = gsl_matrix_complex_column (evec, i);

//        printf ("eigenvalue = %g + %gi\n",
//                GSL_REAL(eval_i), GSL_IMAG(eval_i));
//            printf ("eigenvector = \n");
//            for (j = 0; j < 4; ++j)
//            {
//                gsl_complex z =
//                        gsl_vector_complex_get(&evec_i.vector, j);
//                printf("%g + %gi\n", GSL_REAL(z), GSL_IMAG(z));
//            }

        if (GSL_IMAG(eval_i) == 0.0)
        {
            x.push_back(GSL_REAL(eval_i));
        }
    }

    gsl_vector_complex_free(eval);
    gsl_matrix_complex_free(evec);
}



void p3p(cv::Mat& det_sample, cv::Mat& templ_sample, std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& ts)
{
    cv::Mat det_sample_h = cv::Mat::ones(3, 3, CV_64FC1);
    det_sample.copyTo(det_sample_h(cv::Rect(0,0,3,2)));

    for (int i = 0; i < 3; i++)
    {
//        std::cout << det_sample_h.col(i) <<std::endl;
        double pt_norm = cv::norm(det_sample_h.col(i));
        cv::Mat col_mult = det_sample_h.col(i)*1.0/pt_norm;
        col_mult.copyTo(det_sample_h.col(i));
    }

    double p = 2*det_sample_h.col(1).dot(det_sample_h.col(2));
    double q = 2*det_sample_h.col(0).dot(det_sample_h.col(2));
    double r = 2*det_sample_h.col(0).dot(det_sample_h.col(1));
    double ab = cv::norm(templ_sample.col(0)-templ_sample.col(1));
    double bc = cv::norm(templ_sample.col(1)-templ_sample.col(2));
    double ac = cv::norm(templ_sample.col(0)-templ_sample.col(2));
    double a = (bc/ab)*(bc/ab);
    double b = (ac/ab)*(ac/ab);

    double a0 = -2*b + b*b + a*a + 1 - b*r*r*a + 2*b*a - 2*a;
    double a1 = -2*b*q*a - 2*a*a*q + b*r*r*q*a - 2*q + 2*b*q + 4*a*q + p*b*r + b*r*p*a - b*b*r*p;
    double a2 = q*q + b*b*r*r - b*p*p - q*p*b*r + b*b*p*p - b*r*r*a + 2 - 2*b*b - a*b*r*p*q +
            2*a*a - 4*a - 2*q*q*a + q*q*a*a;
    double a3 = -b*b*r*p + b*r*p*a - 2*a*a*q + q*p*p*b + 2*b*q*a + 4*a*q + p*b*r - 2*b*q - 2*q;
    double a4 = 1 - 2*a + 2*b + b*b - b*p*p + a*a - 2*b*a;

    double z[4];
    double aa[5] = {a0, a1, a2, a3, a4};
    //gsl_poly_complex_workspace * w
//            = gsl_poly_complex_workspace_alloc (5);
//    gsl_poly_complex_solve (aa, 5, w, z);
//    gsl_poly_complex_workspace_free (w);


    //int root_no = quartic(a3/a4, a2/a4, a1/a4, a0/a4, z);
//    int root_no = quartic(a1/a0, a2/a0, a3/a0, a4/a0, z);

    std::vector<double> xs;
//    int root_no = solve_deg4(a0, a1, a2, a3, a4, xs[0], xs[1], xs[2], xs[3]);


    solve_deg4_companion(a0, a1, a2, a3, a4, xs);
//    solve_deg4_companion(1, 0, 0, 0, -1, xs);
    int root_no = xs.size();


//    std::ofstream log_out("/home/alexander/log_p3p");

//    log_out << p << std::endl;
//
//    for (int i = 0; i < 5; i++)
//    {
//        log_out << aa[i] << " ";
//    }
//    log_out << std::endl;
//
//    for (int i = 0; i < root_no; i++)
//    {
//        log_out << z[i] << " ";
//    }
//    log_out << std::endl;

//    xs[0] = 1.03385208447;
//    xs[1] = 0.96194363962;
//    for (int j = 0; j < 2; j ++) {
//        double sum = aa[0];
//        for (int i = 0; i < 4; i++) {
//            sum = sum * xs[j] + aa[i + 1];
//        }
//        std::cout << sum << std::endl;
//    }

    std::vector<double> sols_x, sols_y;
    for (int i = 0; i < root_no; i++)
    {
//        if (fabs(z[2*i+1]) < 1e-5)
        {
            double x = xs[i];//z[2*i];
            double r2 = r*r;
            double s = p*p*a - p*p + b*p*p + p*q*r - q*a*r*p + a*r2 - r2 - b*r2;
            double b0 = b*s*s;
            double r3 = r2*r;
            double r4 = r3*r;
            double r5 = r4*r;
            double p3 = p*p*p;
//            std::cout << s << " " << b0 << std::endl;
            double m1 = ((1 - a - b)*x*x + (q*a - q)*x + 1 - a + b);
            double m2 = (a*a*r3 + 2*b*r3*a - b*r5*a - 2*a*r3 +
                         r3 + pow(b, 2)*r3 - 2*r3*b);
            double m3 = (p*r2 + p*a*a*r2 - 2*b*r3*q*a + 2*r3*b*q -
                         2*r3*q - 2*p*a*r2 - 2*p*r2*b + r4*p*b +
                         4*a*r3*q + b*q*a*r5 - 2*r3*a*a*q +
                         2*r2*p*b*a + b*b*r2*p - r4*p*b*b);
            double m4 = (r3*q*q + r5*b*b + r*p*p*b*b - 4*a*r3 -
                         2*a*r3*q*q + r3*q*q*a*a +
                         2*a*a*r3 - 2*b*b*r3 - 2*p*p*b*r + 4*p*a*r2*q +
                         2*a*p*p*r*b - 2*a*r2*q*b*p - 2*p*p*a*r + r*p*p - b*r5*a + 2*p*r2*b*q +
                         r*p*p*a*a - 2*p*q*r2 + 2*r3 - 2*r2*p*a*a*q - r4*q*b*p);
            double m5 = 4*a*r3*q + p*r2*q*q + 2*p3*b*a - 4*p*a*r2 +
                        -2*r3*b*q - 2*p*p*q*r - 2*b*b*r2*p + r4*p*b + 2*p*a*a*r2
                        - 2*r3*a*a*q - 2*p3*a + p3*a*a + 2*p*r2 + p3 +
                        2*b*r3*q*a+2*q*p*p*b*r + 4*q*a*r*p*p - 2*p*a*r2*q*q - 2*p*p*a*a*r*q +
                        p*a*a*r2*q*q - 2*r3*q - 2*p3*b + p3*b*b - 2*p*p*b*r*q*a;
//            std::cout << m1 << " " << m2 << " " << m3 << " " << m4 << " " << m5 << std::endl;

//            m1 = -0.02577499;
//            m2 = -0.78268143;
//            m3 = 2.34577847;
//            m4 = -2.34250831;
//            m5 = 0.77940992;
//            x = 1.03385208447;
//            b0 = 8.54025651e-09;

//            double pnm = (m2*pow(x, 3) +
//                   m3*x*x +
//                   m4*x +
//                   m5);

            double pnm = m5+x*(m4+x*(m3+x*m2));
            double b1 = m1*pnm;
            double y = b1/b0 ;
//            std::cout << "b0" << std::endl;
//            std::cout << b0 << std::endl;
//            std::cout << "b1" << std::endl;
//            std::cout << b1 << std::endl;

            sols_x.push_back(x);
            sols_y.push_back(y);
//            log_out << x << " " << y << std::endl;
        }
    }

    Rs.clear();
    ts.clear();

//    sols_x.clear();
//    sols_x.push_back(1.03385208);
//    sols_x.push_back(0.96194364 );
//    sols_y.clear();
//    sols_y.push_back(0.98325503);
//    sols_y.push_back(1.01878704);

    for (int si = 0; si < sols_x.size(); si++)
    {
        double x = sols_x[si];
        double y = sols_y[si];
        double v = x*x + y*y - x*y*r;
        double Z = ab / sqrt(v);
        double X = x*Z;
        double Y = y*Z;
        cv::Mat p1c = det_sample_h.col(0)*X;
        cv::Mat p2c = det_sample_h.col(1)*Y;
        cv::Mat p3c = det_sample_h.col(2)*Z;
        cv::Mat coordsc = cv::Mat::zeros(3, 3, CV_64FC1);
        p1c.copyTo(coordsc.col(0));
        p2c.copyTo(coordsc.col(1));
        p3c.copyTo(coordsc.col(2));

//        std::cout << coordsc << std::endl;

        cv::Mat A = cv::Mat::zeros(9, 9, CV_64FC1);
        cv::Mat bm = cv::Mat::zeros(9, 1, CV_64FC1);
        for (int i = 0; i < 3; i++)
        {
            cv::transpose(templ_sample.col(i), A(cv::Rect(0, 3*i, 2, 1)));
            cv::transpose(templ_sample.col(i), A(cv::Rect(2, 3*i+1, 2, 1)));
            cv::transpose(templ_sample.col(i), A(cv::Rect(4, 3*i+2, 2, 1)));
            A.at<double>(3*i, 6) = 1.0;
            A.at<double>(3*i+1, 7) = 1.0;
            A.at<double>(3*i+2, 8) = 1.0;
            coordsc.col(i).copyTo(bm(cv::Rect(0, 3*i, 1, 3)));
        }


//        for (int ay = 0; ay < 9; ay++) {
//            for (int ax = 0; ax < 9; ax++) {
//                log_out << A.at<double>(ay, ax) << " ";
//            }
//            log_out << std::endl;
//        }
//        log_out.flush();

        gsl_matrix_view m = gsl_matrix_view_array((double*)A.data, 9, 9);
        gsl_vector_view b_gsl = gsl_vector_view_array((double*)bm.data, 9);

        gsl_vector *T = gsl_vector_alloc(9);
        int s;
        gsl_permutation *per = gsl_permutation_alloc(9);
        gsl_linalg_LU_decomp(&m.matrix, per, &s);
        gsl_linalg_LU_solve(&m.matrix, per, &b_gsl.vector, T);

//        for (int ti = 0; ti < 9; ti++)
//        {
//            log_out << T->data[ti] << " ";
//        }
//        log_out << std::endl;
//        log_out.flush();

        cv::Mat R = cv::Mat::zeros(3,3,CV_64FC1);
        cv::Mat t(3, 1, CV_64FC1);
        for (int i = 0; i < 3; i ++)
        {
            for (int j = 0; j < 2; j++)
            {
                R.at<double>(i, j) = T->data[2*i+j];
            }
            t.at<double>(i, 0) = T->data[6+i];
        }
//        std::cout << cv::norm(R.col(0)) << std::endl;
        cv::Mat r3 = R.col(0).cross(R.col(1));
        r3.copyTo(R.col(2));
        Rs.push_back(R);
        ts.push_back(t);
    }
}

//
//void test_p3p_2()
//{
//    std::ifstream dets_fin("/home/alexander/materials/carpet/data/p3p_test/dets_p3p");
//    std::ifstream templ_fin("/home/alexander/materials/carpet/data/p3p_test/templ_p3p");
//    cv::Mat dets_mat(2, 3, CV_64FC1);
//    cv::Mat templ_mat(2, 3, CV_64FC1);
//    for (int j = 0; j < 2; j++)
//        for (int i = 0; i < 3; i++)
//
//        {
//            dets_fin >> dets_mat.at<double>(j, i);
//            templ_fin >> templ_mat.at<double>(j, i);
//        }
//    std::vector<cv::Mat> Rs;
//    std::vector<cv::Mat> ts;
//    std::cout << dets_mat << std::endl;
//    p3p(dets_mat, templ_mat, Rs, ts);
//    for (int i = 0; i < Rs.size(); i++)
//    {
//        std::cout << Rs[i] << std::endl;
//    }
//    std::cout << " - " << std::endl;
//}
//
