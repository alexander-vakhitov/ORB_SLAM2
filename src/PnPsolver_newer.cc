/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "PnPsolver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>
#include <opencv/cxeigen.hpp>
#include <opencv2/calib3d.hpp>
#include <include/pnp3d.h>
#include <include/vgl.h>
//#include "triangulate_ceres.h"

using namespace std;

namespace ORB_SLAM2 {

    PnPsolver::PnPsolver(const std::vector<cv::Point2f> &p2D,
                         const std::vector<float> &sigma2, const std::vector<cv::Point3f> &p3D,
                         const std::vector<size_t> &keyPointIndices, const std::vector<size_t> &allIndices,
                         double fu, double fv, double uc, double vc, int nMapPoints,
                         const std::vector<float> &sigmas_3d,
                         const std::vector<Eigen::Matrix3d> &sigmas_3d_full,
                         int mode, bool is_u_ransac, double thrCoeff, bool is_debug_mode, const std::string& debug_path,
                         const std::string& debug_pose_path):
            pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0),
            mnInliersi(0),
            mnIterations(0), mnBestInliers(0), N(0),
            mvP2D(p2D), mvSigma2(sigma2), mvP3Dw(p3D), mvKeyPointIndices(keyPointIndices), mvAllIndices(allIndices),
            fu(fu), fv(fv), uc(uc), vc(vc), nMapPoints(nMapPoints), sigmas_3d(sigmas_3d), sigmas_3d_full(sigmas_3d_full),
            mode(mode),
            N_lines(0),alphas_start(0), alphas_end(0), maximum_number_of_line_correspondences(0), number_of_line_correspondences(0),
            mnRefinedInliersLinesi(0), is_u_ransac(is_u_ransac), thrCoeff (thrCoeff), bUseLines(false),
            is_debug_mode(is_debug_mode), debug_path(debug_path), debug_pose_path(debug_pose_path), filter_bad_3d(true)
    {

        K.setIdentity();
        K(0,0) = fu;
        K(1,1) = fv;
        K(0,2) = uc;
        K(1,2) = vc;


        SetRansacParameters();

        std::cout << " PnP setup: mode " << mode << " unc ransac " << is_u_ransac << " coeff " << thrCoeff << std::endl;
    }



//PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches):
//    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
//    mnIterations(0), mnBestInliers(0), N(0)
//{
//    mvpMapPointMatches = vpMapPointMatches;
//    mvP2D.reserve(F.mvpMapPoints.size());
//    mvSigma2.reserve(F.mvpMapPoints.size());
//    mvP3Dw.reserve(F.mvpMapPoints.size());
//    mvKeyPointIndices.reserve(F.mvpMapPoints.size());
//    mvAllIndices.reserve(F.mvpMapPoints.size());
//
//    int idx=0;
//    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
//    {
//        MapPoint* pMP = vpMapPointMatches[i];
//
//        if(pMP)
//        {
//            if(!pMP->isBad())
//            {
//                const cv::KeyPoint &kp = F.mvKeysUn[i];
//
//                mvP2D.push_back(kp.pt);
//                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);
//
//                cv::Mat Pos = pMP->GetWorldPos();
//                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));
//
//                mvKeyPointIndices.push_back(i);
//                mvAllIndices.push_back(idx);
//
//                idx++;
//            }
//        }
//    }
//
//    // Set camera calibration parameters
//    fu = F.fx;
//    fv = F.fy;
//    uc = F.cx;
//    vc = F.cy;
//
//    SetRansacParameters();
//}

    PnPsolver::~PnPsolver() {
        delete[] pws;
        delete[] us;
        delete[] alphas;
        delete[] pcs;
    }


    void
    PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon,
                                   float th2) {
        mRansacProb = probability;
        mRansacMinInliers = minInliers;
        mRansacMaxIts = maxIterations;
        mRansacEpsilon = epsilon;
        mRansacMinSet = minSet;

        N = mvP2D.size(); // number of correspondences

        mvbInliersi.resize(N);

        // Adjust Parameters according to number of correspondences
        int nMinInliers = N * mRansacEpsilon;
        if (nMinInliers < mRansacMinInliers)
            nMinInliers = mRansacMinInliers;
        if (nMinInliers < minSet)
            nMinInliers = minSet;
        mRansacMinInliers = nMinInliers;

        if (mRansacEpsilon < (float) mRansacMinInliers / N)
            mRansacEpsilon = (float) mRansacMinInliers / N;

        // Set RANSAC iterations according to probability, epsilon, and max iterations
        int nIterations;

        if (mRansacMinInliers == N)
            nIterations = 1;
        else
            nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 3)));

        mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

        mvMaxError.resize(mvSigma2.size());

        for (size_t i = 0; i < mvSigma2.size(); i++)
            mvMaxError[i] = mvSigma2[i] * th2;

        mTh2 = th2;
    }

    cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers) {
        bool bFlag;
        return iterate(mRansacMaxIts, bFlag, vbInliers, nInliers);
    }

    cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
    {
        bNoMore = false;
        vbInliers.clear();
        nInliers = 0;

        set_maximum_number_of_correspondences(mRansacMinSet);

        set_maximum_number_of_line_correspondences(0);

        reset_line_correspondences();

        if (N < mRansacMinInliers) {
            bNoMore = true;
            return cv::Mat();
        }

        vector<size_t> vAvailableIndices;

        int nCurrentIterations = 0;
        while (mnIterations < mRansacMaxIts || nCurrentIterations < nIterations) {
            nCurrentIterations++;
            mnIterations++;
            reset_correspondences();

            vAvailableIndices = mvAllIndices;

            // Get min set of points
            for (short i = 0; i < mRansacMinSet; ++i) {
                int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

                int idx = vAvailableIndices[randi];

                add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y, 0, 0, Eigen::Matrix3d::Identity());

                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }

            // Compute camera pose
            compute_pose(mRi, mti);

            // Check inliers
            CheckInliers();



//        std::cout << " ransac inliers " << mnInliersi << " need " << mRansacMinInliers << std::endl;

            if (mnInliersi >= mRansacMinInliers) {
                // If it is the best solution so far, save it
                if (mnInliersi > mnBestInliers) {
                    mvbBestInliers = mvbInliersi;
                    mnBestInliers = mnInliersi;

                    cv::Mat Rcw(3, 3, CV_64F, mRi);
                    cv::Mat tcw(3, 1, CV_64F, mti);
                    Rcw.convertTo(Rcw, CV_32F);
                    tcw.convertTo(tcw, CV_32F);
                    mBestTcw = cv::Mat::eye(4, 4, CV_32F);
                    Rcw.copyTo(mBestTcw.rowRange(0, 3).colRange(0, 3));
                    tcw.copyTo(mBestTcw.rowRange(0, 3).col(3));
                }

                if (bUseLines) {
                    CheckInliersLines();
                }

                if (Refine()) {
//                    std::cout << " refined inliers " << mnRefinedInliers << " lines " << mnRefinedInliersLinesi << std::endl;

                    nInliers = mnRefinedInliers;
                    vbInliers = vector<bool>(nMapPoints, false);
                    for (int i = 0; i < N; i++) {
                        if (mvbRefinedInliers[i])
                            vbInliers[mvKeyPointIndices[i]] = true;
                    }
                    return mRefinedTcw.clone();
                }

            }
        }

        if (mnIterations >= mRansacMaxIts) {
            bNoMore = true;
            if (mnBestInliers >= mRansacMinInliers) {
                nInliers = mnBestInliers;
                vbInliers = vector<bool>(nMapPoints, false);
                for (int i = 0; i < N; i++) {
                    if (mvbBestInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mBestTcw.clone();
            }
        }

        return cv::Mat();
    }

    void PnPsolver::SwitchMode(int new_mode)
    {
        this->mode = mode;
    }

    cv::Mat PnPsolver::FinalRefinement(const cv::Mat& T_init, std::vector<bool>& vbInliers, int& nInliers)
    {
//        mvbBestInliers = inliersInit;
        mBestTcw = T_init.clone();

        if (bUseLines)
        {
            CheckInliersLines();
            std::cout << " line inliers checked " << std::endl;
        }

        if (Refine())
        {
            nInliers = mnRefinedInliers;
            vbInliers = vector<bool>(nMapPoints, false);
            for (int i = 0; i < N; i++) {
                if (mvbRefinedInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }

            return mRefinedTcw;
        } else {
            return T_init;
        }
    }

    void PnPsolver::AddLineCorrespondences()
    {
        vector<int> lineIndices;
//        lineIndices.reserve(N_lines);
        for (int i = 0; i < N_lines; i++)
        {
            if (mvbInliersLines[i])
            {
                lineIndices.push_back(i);
            }
        }

//        std::cout << " max line corr num " << lineIndices.size() << std::endl;

        set_maximum_number_of_line_correspondences(lineIndices.size());

        reset_line_correspondences();

        Eigen::Matrix4d T_est;
        cv::cv2eigen(mBestTcw, T_est);

        for (int i = 0; i < lineIndices.size(); i++)
        {
            int idx = lineIndices[i];
            Eigen::Matrix3d sigma2d, sigma2dn;
            if (idx < mvSigmasLines2D.size())
            {
                sigma2d = mvSigmasLines2D[idx];
            } else {
                std::cout << " error s2d " << std::endl;
            }
            if (idx < mvSigmasLines2D.size())
            {
                sigma2dn = mvSigmasLines2DNorm[idx];
            } else {
                std::cout << " error s2dn " << std::endl;
            }
            Eigen::Matrix<double, 6, 6> sigma6d;
            if (idx < mvSigmasLines3D.size())
            {
                sigma6d = mvSigmasLines3D[idx];
            } else {
                std::cout << " error s6d " << std::endl;
            }
//            std::cout << " adding line " << idx  << " all line eq num " << mvLineEqs.size() << std::endl;
            if (idx>=mvStartPts.size() || idx >= mvEndPts.size() || idx >= mvLineEqs.size())
            {
                std::cout << " error getting line data" << std::endl;
            }
            Eigen::Vector3d Xs0 = mvStartPts[idx];
            Eigen::Vector3d Xe0 = mvEndPts[idx];
            Eigen::Vector3d line_dir = mvLineDirs[idx];
            Eigen::Vector3d X0 = mvLineX0s[idx];// - line_dir.dot(Xs0) * line_dir;

            Eigen::Vector3d XScorr, XEcorr;
            vgl::ReprojectEndpointTo3DProjMat(K*T_est.block<3,4>(0,0), mvLineEndpts2dStart[idx], X0, line_dir, &XScorr);
            vgl::ReprojectEndpointTo3DProjMat(K*T_est.block<3,4>(0,0), mvLineEndpts2dEnd[idx], X0, line_dir, &XEcorr);
            double p_start = XScorr.dot(line_dir);
            double p_fin = XEcorr.dot(line_dir);

//            p_start = 0;
//            p_fin = 1;
            Eigen::Matrix<double,6,6> SigmaFin;
//            SigmaFin.block<3,3>(0,0) = CovLine3DPoints(sigma6d, p_start, p_start);
//            SigmaFin.block<3,3>(0,3) = CovLine3DPoints(sigma6d, p_start, p_fin);
//            SigmaFin.block<3,3>(3,0) = CovLine3DPoints(sigma6d, p_fin, p_start);
//            SigmaFin.block<3,3>(3,3) = CovLine3DPoints(sigma6d, p_fin, p_fin);

            add_line_correspondence(sigma2d, sigma2dn, sigma6d, Xs0, Xe0, mvLineEqs[idx],
                    mvLineEqsNorm[idx], mvLineEndpts2dStart[idx], mvLineEndpts2dEnd[idx], mvMaxLineError[idx]/mTh2,
                    sigma6d.block<3,3>(3,3));

//simple
//            add_line_correspondence(sigma2d, sigma2dn, sigma6d, X0, X0+line_dir, mvLineEqs[idx],
//                    mvLineEqsNorm[idx], mvLineEndpts2dStart[idx], mvLineEndpts2dEnd[idx], mvMaxLineError[idx]/mTh2,
//                    sigma6d.block<3,3>(3,3));
        }

//        std::cout << " added line corrs " << std::endl;
    }

    bool PnPsolver::Refine() {
        vector<int> vIndices;
        vIndices.reserve(mvbBestInliers.size());

        for (size_t i = 0; i < mvbBestInliers.size(); i++) {
            if (mvbBestInliers[i]) {
                vIndices.push_back(i);
            }
        }

        set_maximum_number_of_correspondences(vIndices.size());

        reset_correspondences();

        open_log();

        for (size_t i = 0; i < vIndices.size(); i++) {
            int idx = vIndices[i];
            float sigma2d = 0.0;
            float sigma3d = 0.0;
            if (idx < mvSigma2.size()) {
                sigma2d = mvSigma2[idx];
            }
            if (idx < sigmas_3d.size()) {
                sigma3d = sigmas_3d[idx];
            }
            Eigen::Matrix3d S3D = Eigen::Matrix3d::Zero();
            if (idx < sigmas_3d_full.size())
            {
                S3D = sigmas_3d_full[idx];
            } else {
                std::cout << " error s3d " << idx << " " << sigmas_3d_full.size() << " " << mvbBestInliers.size() << std::endl;
            }
            add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y, sigma2d,
                               sigma3d, S3D);
        }

        finish_log();

//        std::cout << " added point corrs " << std::endl;

        if (bUseLines)
        {
            AddLineCorrespondences();
        }

//        std::cout << " before pose, mode = " << mode << std::endl;

        // Compute camera pose
        if (mode == 0) {
            compute_pose(mRi, mti);
        }
        if (mode == 1)
        {
            compute_pose_uncertain(mBestTcw, mRi, mti);
        }
        if (mode == 2)
        {
            compute_pose_dlsu(mBestTcw, mRi, mti);
        }
        if (mode == 3)
        {
            compute_pose_dlsu(mBestTcw, mRi, mti, true);
        }

//        std::cout << " after pose " << std::endl;
        // Check inliers
        CheckInliers();

        if (bUseLines)
        {
            CheckInliersLines();
            mnRefinedInliersLinesi = mnInliersLinesi;
        }


//    std::cout << " pnp inliers " << mnInliersi << std::endl;

        mnRefinedInliers = mnInliersi;
        mvbRefinedInliers = mvbInliersi;


        if (mnInliersi > mRansacMinInliers) {
            cv::Mat Rcw(3, 3, CV_64F, mRi);
            cv::Mat tcw(3, 1, CV_64F, mti);
            Rcw.convertTo(Rcw, CV_32F);
            tcw.convertTo(tcw, CV_32F);
            mRefinedTcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(mRefinedTcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(mRefinedTcw.rowRange(0, 3).col(3));
            return true;
        }

        return false;
    }

    void PnPsolver::CheckInliersLines()
    {
        mnInliersLinesi = 0;

        mvbInliersLines = std::vector<bool>(mvLineEndpts2dEnd.size(), false);

//        return;

        Eigen::Matrix3d R_est;
        Eigen::Vector3d t_est;
        for (int ri = 0; ri < 3; ri++)
        {
            for (int ci = 0; ci < 3; ci++)
            {
                R_est(ri,ci) = mRi[ri][ci];
            }
            t_est(ri) = mti[ri];
        }
        PMatrix P;
        P.block<3,3>(0,0) = R_est;
        P.block<3,1>(0,3) = t_est;
        for (int i = 0; i < N_lines; i++)
        {
            Eigen::Vector3d Xs_c = K*(R_est*mvStartPts[i] + t_est);
            Eigen::Vector3d xs_h = Xs_c/Xs_c(2);
            Eigen::Vector3d Xe_c = K*(R_est*(mvEndPts[i]) + t_est);
            Eigen::Vector3d xe_h = Xe_c/Xe_c(2);

            Eigen::Vector3d l_h = xs_h.cross(xe_h);
            l_h = l_h/l_h.segment<2>(0).norm();
            const Eigen::Vector2d& xs = mvLineEndpts2dStart[i];
            const Eigen::Vector2d& xe = mvLineEndpts2dEnd[i];
//            double errStart = (xs-xs_h.segment<2>(0)).norm();
//            double errEnd = (xe-xe_h.segment<2>(0)).norm();
            double errStart = l_h.dot(Eigen::Vector3d(xs(0), xs(1), 1.0));
            double errEnd = l_h.dot(Eigen::Vector3d(xe(0), xe(1), 1.0));
//            Eigen::Vector3d l_det = Eigen::Vector3d(xs(0), xs(1), 1.0).cross(Eigen::Vector3d(xe(0), xe(1), 1.0));
//            l_det = l_det / l_det.segment<2>(0).norm();

//            double errStart = l_det.dot(xs_h);
//            double errEnd = l_det.dot(xe_h);
            //ReprojectEndpointTo3DProjMat(const PMatrix& P, const Eigen::Vector2d& endpoint, const Eigen::Vector3d& X0,
            // const Eigen::Vector3d& line_dir, Eigen::Vector3d* endpoint_3d);
            Eigen::Vector3d eptStart3d, eptEnd3d;
            vgl::ReprojectEndpointTo3DProjMat(P, xs, mvLineX0s[i], mvLineDirs[i], &eptStart3d);
            vgl::ReprojectEndpointTo3DProjMat(P, xe, mvLineX0s[i], mvLineDirs[i], &eptEnd3d);
            double error2 = errStart*errStart + errEnd*errEnd;
            if (error2 < mvMaxLineError[i] && eptStart3d(2) > 0 && eptEnd3d(2) > 0)
            {
                mvbInliersLines[i] = true;
                mnInliersLinesi++;
            } else {
                mvbInliersLines[i] = false;
            }
        }
    }

    void PnPsolver::SetLines(const std::vector<Eigen::Vector3d>& linesStartPts,
            const std::vector<Eigen::Vector3d>& linesEndPts,
            const std::vector<Eigen::Vector3d>& X0s, const std::vector<Eigen::Vector3d>& lineDirs,
             const vec2d& line_endpts_2d_start,
             const vec2d& line_endpts_2d_end,
            const mat6dvector& linesSigmas3D,
            const std::vector<Eigen::Matrix3d>& linesSigmasProj,
            const std::vector<Eigen::Matrix3d>& linesSigmasProjNorm,
            const std::vector<float>& linesDetSigmas2)
    {
        bUseLines = true;
        N_lines = linesStartPts.size();
        mvStartPts = linesStartPts;
        mvEndPts = linesEndPts;
        mvLineEndpts2dStart = line_endpts_2d_start;
        mvLineEndpts2dEnd = line_endpts_2d_end;
//        mvLineEqs = linesProjEqs;
        mvSigmasLines3D = linesSigmas3D;
        mvSigmasLines2D = linesSigmasProj;
        mvSigmasLines2DNorm = linesSigmasProjNorm;
        mvMaxLineError.resize(N_lines);
        mvLineDirs = lineDirs;
        mvLineX0s = X0s;
        mvLineEqs.resize(N_lines);
        mvLineEqsNorm.resize(N_lines);

        for (int i = 0; i < N_lines; i++)
        {
            mvMaxLineError[i] = linesDetSigmas2[i] * mTh2 * mTh2 ;
//            Eigen::Vector3d dLine = mvStartPts[i]-mvEndPts[i];
//            mvLineDirs[i] = dLine/dLine.norm();
//            mvLineX0s[i] = mvStartPts[i] - (mvLineDirs[i].dot(mvStartPts[i]))*mvLineDirs[i];
            Eigen::Vector3d xsh(mvLineEndpts2dStart[i](0), mvLineEndpts2dStart[i](1), 1.0);
            Eigen::Vector3d xeh(mvLineEndpts2dEnd[i](0), mvLineEndpts2dEnd[i](1), 1.0);
            Eigen::Vector3d lineEq2d = xsh.cross(xeh);
            lineEq2d = lineEq2d / lineEq2d.segment<2>(0).norm();
            mvLineEqs[i] = lineEq2d;
            Eigen::Vector3d lineEq2dn = K.transpose() * lineEq2d;
            lineEq2dn = lineEq2dn/lineEq2dn.segment<2>(0).norm();
            mvLineEqsNorm[i] = lineEq2dn;
        }
        mvbInliersLines.resize(N_lines);
        maximum_number_of_line_correspondences = 0;
    }

    void GetJProj(double Xc, double Yc, double Zc, Eigen::Matrix<double, 2, 3>* J_proj_p)
    {
        Eigen::Matrix<double, 2, 3> J_proj;
        J_proj.setZero();
        J_proj(0, 0) = 1.0 / Zc;
        J_proj(1, 1) = 1.0 / Zc;
        J_proj(0, 2) = -Xc / Zc / Zc;
        J_proj(1, 2) = -Yc / Zc / Zc;
        *J_proj_p = J_proj;
    }

    void PnPsolver::CheckInliers() {

        Eigen::Matrix3d R_est;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                R_est(i, j) = mRi[i][j];
            }
        }
        Eigen::Matrix2d F;
        F.setZero();
        F(0, 0) = fu ;
        F(1, 1) = fv ;

        mnInliersi = 0;

        for (int i = 0; i < N; i++) {
            cv::Point3f P3Dw = mvP3Dw[i];
            cv::Point2f P2D = mvP2D[i];

            float Xc = mRi[0][0] * P3Dw.x + mRi[0][1] * P3Dw.y + mRi[0][2] * P3Dw.z + mti[0];
            float Yc = mRi[1][0] * P3Dw.x + mRi[1][1] * P3Dw.y + mRi[1][2] * P3Dw.z + mti[1];
            float invZc = 1 / (mRi[2][0] * P3Dw.x + mRi[2][1] * P3Dw.y + mRi[2][2] * P3Dw.z + mti[2]);

            double ue = uc + fu * Xc * invZc;
            double ve = vc + fv * Yc * invZc;

            float distX = P2D.x - ue;
            float distY = P2D.y - ve;

            Eigen::Matrix2d SigmaProj;
            SigmaProj.setZero();
            bool is_cov_pnp = (mode > 0 && mode < 4);
            Eigen::Matrix2d SigmaDet = Eigen::Matrix2d::Identity() * mvSigma2[i];
            Eigen::Matrix2d SigmaFull = SigmaDet;
            Eigen::Matrix<double, 2, 3> J_proj;
            GetJProj(Xc, Yc, 1.0/invZc, &J_proj);
            SigmaProj = F * J_proj * R_est * sigmas_3d_full[i] * R_est.transpose() * J_proj.transpose() * F;

            if (is_cov_pnp && is_u_ransac) {
                SigmaFull = SigmaFull + SigmaProj;
//                std::cout << sigmas_3d_full[i] << " " << SigmaProj << " " << mvSigma2[i] << std::endl;
//                std::cout << " - " << std::endl;
            }

//            if (SigmaProj.trace() > 100 * SigmaDet.trace())
            if (filter_bad_3d && is_cov_pnp && SigmaProj.trace() > 40)
            {
                mvbInliersi[i] = false;
                continue;
            }

            Eigen::Matrix2d S = SigmaFull.inverse();

            Eigen::Vector2d dP;
            dP(0) = distX;
            dP(1) = distY;
            float error2 = dP.transpose() * S * dP;
            float error2_3d = dP.transpose() * SigmaProj.inverse() * dP;

            if ((error2 < thrCoeff * mTh2)  && invZc > 0) // && (!is_cov_pnp || error2_3d < thrCoeff * mTh2)
            {
                mvbInliersi[i] = true;
                mnInliersi++;
            } else {
                mvbInliersi[i] = false;
            }


//            float error2 = distX * distX + distY * distY;

//            if (error2 < mvMaxError[i] && invZc > 0) {
//                mvbInliersi[i] = true;
//                mnInliersi++;
//            } else {
//                mvbInliersi[i] = false;
//            }
        }
    }


    void PnPsolver::set_maximum_number_of_correspondences(int n) {
        if (maximum_number_of_correspondences < n) {
            if (pws != 0) delete[] pws;
            if (us != 0) delete[] us;
            if (alphas != 0) delete[] alphas;
            if (pcs != 0) delete[] pcs;

            maximum_number_of_correspondences = n;
            pws = new double[3 * maximum_number_of_correspondences];
            us = new double[2 * maximum_number_of_correspondences];
            alphas = new double[4 * maximum_number_of_correspondences];
            pcs = new double[3 * maximum_number_of_correspondences];
        }

        sigmas2d_selected = std::vector<float>(maximum_number_of_correspondences, 0);

        sigmas3d_selected = std::vector<float>(maximum_number_of_correspondences, 0);

        Sigmas3D_selected = std::vector<Eigen::Matrix3d>(maximum_number_of_correspondences, Eigen::Matrix3d::Identity());
    }

    void PnPsolver::set_maximum_number_of_line_correspondences(int n) {
        if (maximum_number_of_line_correspondences < n) {
//            if (pws != 0) delete[] pws;
//            if (us != 0) delete[] us;
            if (alphas_start != 0) delete[] alphas_start;
            if (alphas_end != 0) delete[] alphas_end;
//            if (pcs != 0) delete[] pcs;

            maximum_number_of_line_correspondences = n;

            alphas_start = new double[4 * maximum_number_of_line_correspondences];
            alphas_end = new double[4 * maximum_number_of_line_correspondences];
        }

        Sigmas2dlines_selected = std::vector<Eigen::Matrix3d>(maximum_number_of_line_correspondences, Eigen::Matrix3d::Zero());
        Sigmas2dlinesN_selected = std::vector<Eigen::Matrix3d>(maximum_number_of_line_correspondences, Eigen::Matrix3d::Zero());
        Sigmas3dlines_selected = mat6dvector(maximum_number_of_line_correspondences,
                Eigen::Matrix<double,6,6>::Identity());

        Xs_selected = std::vector<Eigen::Vector3d>(maximum_number_of_line_correspondences, Eigen::Vector3d::Zero());
        Xe_selected = std::vector<Eigen::Vector3d>(maximum_number_of_line_correspondences, Eigen::Vector3d::Zero());
        lineEqs_selected = std::vector<Eigen::Vector3d>(maximum_number_of_line_correspondences, Eigen::Vector3d::Zero());
        lineEqsN_selected = std::vector<Eigen::Vector3d>(maximum_number_of_line_correspondences, Eigen::Vector3d::Zero());
        start_pts_2d_selected = vec2d(maximum_number_of_line_correspondences, Eigen::Vector2d::Zero());
        end_pts_2d_selected = vec2d(maximum_number_of_line_correspondences, Eigen::Vector2d::Zero());
        sigmasDetLineSelected = std::vector<float>(maximum_number_of_line_correspondences, 0);
        Sigmas_LD_selected = std::vector<Eigen::Matrix3d>(maximum_number_of_line_correspondences, Eigen::Matrix3d::Zero());
    }

    void PnPsolver::reset_correspondences(void) {
        number_of_correspondences = 0;
    }

    void PnPsolver::reset_line_correspondences(void)
    {
        number_of_line_correspondences = 0;
    }

    void PnPsolver::finish_log()
    {
        if (is_debug_mode)
        {
            debug_output.close();
            is_debug_opened = false;
            output_initial_pose();
        }
    }


    void PnPsolver::output_initial_pose()
    {
        if (is_debug_mode)
        {
            std::ofstream pose_out(debug_pose_path);
            for (int ii = 0; ii < 4; ii++)
            {
                for (int jj = 0; jj < 4; jj++)
                {
                    pose_out << mBestTcw.at<float>(ii, jj) << " " ;
                }
            }
            pose_out << std::endl;
        }
    }

    void PnPsolver::open_log()
    {
        if (is_debug_mode)
        {
            debug_output = std::ofstream(debug_path);
            is_debug_opened = true;
        }
    }

    void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v, const double s2d,
                                       const double s3d, const Eigen::Matrix3d& S3D) {
        pws[3 * number_of_correspondences] = X;
        pws[3 * number_of_correspondences + 1] = Y;
        pws[3 * number_of_correspondences + 2] = Z;

        us[2 * number_of_correspondences] = u;
        us[2 * number_of_correspondences + 1] = v;

        sigmas2d_selected[number_of_correspondences] = s2d;
        sigmas3d_selected[number_of_correspondences] = s3d;
        Sigmas3D_selected[number_of_correspondences] = S3D;

        if (s3d < 0.001)
        {
            double c = 0.001 / s3d;
            sigmas3d_selected[number_of_correspondences] *= c;
            Sigmas3D_selected[number_of_correspondences] = c * Sigmas3D_selected[number_of_correspondences];
        }

        if (s3d > 1)
        {
            double c = 1.0 / s3d;
            sigmas3d_selected[number_of_correspondences] *= c;
            Sigmas3D_selected[number_of_correspondences] = c * Sigmas3D_selected[number_of_correspondences];
        }

        if (is_debug_opened)
        {
            debug_output << X << " " << Y << " " << Z << " " << u << " " << v << " " << s2d << " " << s3d << " ";
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    debug_output << S3D(i, j) << " ";
                }
            }
            debug_output << std::endl;
        }


        number_of_correspondences++;
    }

//    void PnPsolver::fill_M_uncertain_line(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, CvMat *M,
//                                          const int row, const double *as, const double *ae,
//                                          const Eigen::Vector3d& line_eq, const Eigen::Matrix3d& s2d,
//                                          const Eigen::Matrix<double,6,6>&S6D, const Eigen::Matrix3d& K,
//                                          const Eigen::Vector3d& Xs, const Eigen::Vector3d& Xe)

    void PnPsolver::add_line_correspondence(const Eigen::Matrix3d& S2d, const Eigen::Matrix3d& S2dN,
            const Eigen::Matrix<double,6,6>& S6D,
            const Eigen::Vector3d& X_start, const Eigen::Vector3d& X_end,
            const Eigen::Vector3d& lineEq, const Eigen::Vector3d& lineEqN,
            const Eigen::Vector2d& xs, const Eigen::Vector2d& xe, float detSigma2,
            const Eigen::Matrix3d& SigmaLD)
    {
        if (number_of_line_correspondences >= Sigmas2dlines_selected.size() ||
                number_of_line_correspondences >= Sigmas3dlines_selected.size()  ||
                number_of_line_correspondences >= Xs_selected.size() ||
                number_of_line_correspondences >= Xe_selected.size() ||
                number_of_line_correspondences >= lineEqs_selected.size() )
        {
            std::cout << " error adding line correspondence" << std::endl;
        }
        Sigmas2dlines_selected[number_of_line_correspondences] = S2d;
        Sigmas2dlinesN_selected[number_of_line_correspondences] = S2dN;
        Sigmas3dlines_selected[number_of_line_correspondences] = S6D;
        Xs_selected[number_of_line_correspondences] = X_start;
        Xe_selected[number_of_line_correspondences] = X_end;
        lineEqs_selected[number_of_line_correspondences] = lineEq;
        lineEqsN_selected[number_of_line_correspondences] = lineEqN;
        start_pts_2d_selected[number_of_line_correspondences] = xs;
        end_pts_2d_selected[number_of_line_correspondences] = xe;
        sigmasDetLineSelected[number_of_line_correspondences] = detSigma2;
        Sigmas_LD_selected[number_of_line_correspondences] = SigmaLD;
        number_of_line_correspondences++;
    }

    void PnPsolver::choose_control_points(void) {
        // Take C0 as the reference points centroid:
        cws[0][0] = cws[0][1] = cws[0][2] = 0;
        for (int i = 0; i < number_of_correspondences; i++)
            for (int j = 0; j < 3; j++)
                cws[0][j] += pws[3 * i + j];

        if (bUseLines) {
//            std::cout << " choosing cp using lines " << std::endl;
            for (int i = 0; i < number_of_line_correspondences; i++) {
//                std::cout << Xs_selected[i].transpose() << " " << Xe_selected[i].transpose() << std::endl;
                for (int j = 0; j < 3; j++) {
                    cws[0][j] += Xs_selected[i](j);
                    cws[0][j] += Xe_selected[i](j);
                }
            }
        }

        int n = number_of_correspondences;
        if (bUseLines)
        {
            n += 2* number_of_line_correspondences;
        }

        for (int j = 0; j < 3; j++)
            cws[0][j] /= n;


        // Take C1, C2, and C3 from PCA on the reference points:
        CvMat *PW0 = cvCreateMat(n, 3, CV_64F);

        double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
        CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
        CvMat DC = cvMat(3, 1, CV_64F, dc);
        CvMat UCt = cvMat(3, 3, CV_64F, uct);

        for (int i = 0; i < number_of_correspondences; i++)
            for (int j = 0; j < 3; j++)
                PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

        if (bUseLines)
        {
            for (int i = 0; i < number_of_line_correspondences; i++)
                for (int j = 0; j < 3; j++) {
                    PW0->data.db[3*number_of_correspondences+3 * i + j] = Xs_selected[i](j) - cws[0][j];
                    PW0->data.db[3*(number_of_correspondences+number_of_line_correspondences) + 3 * i + j] = Xe_selected[i](j) - cws[0][j];
                }

        }

        cvMulTransposed(PW0, &PW0tPW0, 1);
        cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

        cvReleaseMat(&PW0);

        for (int i = 1; i < 4; i++) {
            double k = sqrt(dc[i - 1] / n);
            for (int j = 0; j < 3; j++)
                cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
        }
    }

    void PnPsolver::choose_control_points_uncertain(void) {
        // Take C0 as the reference points centroid:
        cws[0][0] = cws[0][1] = cws[0][2] = 0;

        double sigma_inv_sum = 0.0;
        for (int i = 0; i < number_of_correspondences; i++) {
            double sigma_inv = 1.0 / (sigmas3d_selected[i] + 1e-6);
            sigma_inv_sum += sigma_inv;
            for (int j = 0; j < 3; j++)
                cws[0][j] += sigma_inv * pws[3 * i + j];
        }

//        std::vector<double> sigmas_start, sigmas_end;
        std::vector<double> sigmas_lines;

        if (bUseLines)
        {
//            for (int i = 0; i < number_of_line_correspondences; i++) {
//                const Eigen::Matrix<double,6,6>& S6D = Sigmas3dlines_selected[i];
//                Eigen::Matrix3d SigmaLinesSum = S6D.block<3,3>(0,0) + S6D.block<3,3>(0,3) + S6D.block<3,3>(3,0) + S6D.block<3,3>(3,3);
////                double sigma_inv_line = 6.0/S6D.trace();
////                double sigma_inv_end = 6.0/S6D.trace();
////                sigmas_start.push_back(1.0/sigma_inv_start);
////                sigmas_end.push_back(1.0/sigma_inv_end);
//                double sigma_line = 0.25*SigmaLinesSum.trace()*1.0/3.0;
//                sigmas_lines.push_back(sigma_line);
//                sigma_inv_sum += sigma_line;
//                for (int j = 0; j < 3; j++)
//                    cws[0][j] += 1.0/sigma_line* 0.5*(Xs_selected[i](j) + Xe_selected[i](j) );
//            }
        }

        double sigma_sum = 1.0 / (sigma_inv_sum + 1e-6);

        for (int j = 0; j < 3; j++) {
            cws[0][j] = sigma_sum * cws[0][j];
        }

        int n = number_of_correspondences;
        if (bUseLines)
        {
//            n += number_of_line_correspondences;
        }
        // Take C1, C2, and C3 from PCA on the reference points:
        CvMat *PW0 = cvCreateMat(n, 3, CV_64F);

        double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
        CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
        CvMat DC = cvMat(3, 1, CV_64F, dc);
        CvMat UCt = cvMat(3, 3, CV_64F, uct);

        for (int i = 0; i < number_of_correspondences; i++) {
            double sigma_remain = sigmas3d_selected[i] - sigma_sum + 1e-6;
            for (int j = 0; j < 3; j++)
                PW0->data.db[3 * i + j] = 1.0 / sqrt(fabs(sigma_remain)) * (pws[3 * i + j] - cws[0][j]);
        }
        if (bUseLines)
        {
//            for (int i = 0; i < number_of_line_correspondences; i++) {
//                double sigma_remain_line = sigmas_lines[i] - sigma_sum + 1e-6;
//                for (int j = 0; j < 3; j++) {
//                    PW0->data.db[3*number_of_correspondences + 3 * i + j] = 1.0 / sqrt(fabs(sigma_remain_line)) * ( 0.5*(Xs_selected[i](j) + Xe_selected[i](j)) - cws[0][j]);
//                }
//            }
        }

        cvMulTransposed(PW0, &PW0tPW0, 1);
        cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

        cvReleaseMat(&PW0);

        for (int i = 1; i < 4; i++) {
            double k = sqrt(dc[i - 1] / n);
            for (int j = 0; j < 3; j++)
                cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
        }
    }

    void PnPsolver::choose_control_points_uncertain2(void) {
        // Take C0 as the reference points centroid:
        cws[0][0] = cws[0][1] = cws[0][2] = 0;

        double sigma_inv_sum = 0.0;
        for (int i = 0; i < number_of_correspondences; i++) {
            double sigma_inv = 1.0 / (sigmas3d_selected[i] + 1e-6);
            sigma_inv_sum += sigma_inv;
            for (int j = 0; j < 3; j++)
                cws[0][j] += sigma_inv * pws[3 * i + j];
        }

//        std::vector<double> sigmas_start, sigmas_end;
        std::vector<Eigen::Matrix3d> sigmas_lines;
        Eigen::Matrix3d SigmasLinesInvSum;
        SigmasLinesInvSum.setZero();

        if (bUseLines)
        {
            for (int i = 0; i < number_of_line_correspondences; i++) {
                const Eigen::Matrix<double,6,6>& S6D = Sigmas3dlines_selected[i];
                Eigen::Matrix3d SigmaLinesSum = S6D.block<3,3>(0,0) + S6D.block<3,3>(0,3) + S6D.block<3,3>(3,0) + S6D.block<3,3>(3,3);
//                double sigma_inv_line = 6.0/S6D.trace();
//                double sigma_inv_end = 6.0/S6D.trace();
//                sigmas_start.push_back(1.0/sigma_inv_start);
//                sigmas_end.push_back(1.0/sigma_inv_end);
//                double sigma_line = SigmaLinesSum.trace()*1.0/3.0;
                sigmas_lines.push_back(SigmaLinesSum);
                SigmasLinesInvSum += S6D.block<3,3>(0,0).inverse() + S6D.block<3,3>(3,3).inverse();
                Eigen::Vector3d dX = S6D.block<3,3>(0,0).inverse()* Xs_selected[i] + S6D.block<3,3>(3,3).inverse() * Xe_selected[i] ;
                for (int j = 0; j < 3; j++)
                    cws[0][j] += dX(j);
            }
        }

        Eigen::Matrix3d SigmaSum = (SigmasLinesInvSum + sigma_inv_sum*Eigen::Matrix3d::Identity()).inverse();
        Eigen::Vector3d Xm(cws[0][0], cws[0][1], cws[0][2]);
        Eigen::Vector3d Xmc = SigmaSum*Xm;
//        double sigma_sum = 1.0 / (sigma_inv_sum + 1e-6);

        for (int j = 0; j < 3; j++) {
            cws[0][j] = Xmc(j);
        }

        int n = number_of_correspondences;
        if (bUseLines)
        {
            n += 2*number_of_line_correspondences;
        }
        // Take C1, C2, and C3 from PCA on the reference points:
        CvMat *PW0 = cvCreateMat(n, 3, CV_64F);

        double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
        CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
        CvMat DC = cvMat(3, 1, CV_64F, dc);
        CvMat UCt = cvMat(3, 3, CV_64F, uct);

        for (int i = 0; i < number_of_correspondences; i++) {
            Eigen::Matrix3d sigma_remain = (sigmas3d_selected[i]+1e-6)*Eigen::Matrix3d::Identity() - SigmaSum ;

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(sigma_remain, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector3d svals = svd.singularValues();
            Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
            S(0,0) = 1.0/sqrt(fabs(svals(0)));
            S(1,1) = 1.0/sqrt(fabs(svals(1)));
            S(2,2) = 1.0/sqrt(fabs(svals(2)));
            Eigen::Matrix3d Sigma3dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

            Eigen::Vector3d Xp(pws[3*i], pws[3*i+1], pws[3*i+2]);
            Eigen::Vector3d Xpc = Sigma3dInv_sqrt * (Xp - Xmc);
            for (int j = 0; j < 3; j++)
                PW0->data.db[3 * i + j] = Xpc(j);

        }
        if (bUseLines)
        {
            for (int i = 0; i < number_of_line_correspondences; i++) {
                const Eigen::Matrix<double, 6, 6> &S6D = Sigmas3dlines_selected[i];
                for (int k = 0; k < 2; k++) {
                    Eigen::Matrix3d sigma_remain =
                            S6D.block<3, 3>(3 * k, 3 * k) + 1e-6 * Eigen::Matrix3d::Identity() - SigmaSum;

                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(sigma_remain, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Vector3d svals = svd.singularValues();
                    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
                    S(0, 0) = 1.0 / sqrt(fabs(svals(0)));
                    S(1, 1) = 1.0 / sqrt(fabs(svals(1)));
                    S(2, 2) = 1.0 / sqrt(fabs(svals(2)));
                    Eigen::Matrix3d Sigma3dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

                    Eigen::Vector3d Xp;
                    if (k == 0)
                    {
                        Xp = Xs_selected[i];
                    } else {
                        Xp = Xe_selected[i];
                    }
                    Eigen::Vector3d Xpc = Sigma3dInv_sqrt * (Xp - Xmc);
                    for (int j = 0; j < 3; j++)
                        PW0->data.db[3 *(number_of_correspondences + k*number_of_line_correspondences + i) + j] = Xpc(j);
                }
            }
        }

        cvMulTransposed(PW0, &PW0tPW0, 1);
        cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

        cvReleaseMat(&PW0);

        for (int i = 1; i < 4; i++) {
            double k = sqrt(dc[i - 1] / n);
            for (int j = 0; j < 3; j++)
                cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
        }
    }

//    void PnPsolver::choose_control_points_uncertain(void) {
//        // Take C0 as the reference points centroid:
//        cws[0][0] = cws[0][1] = cws[0][2] = 0;
//
//        double sigma_inv_sum = 0.0;
//        for (int i = 0; i < number_of_correspondences; i++) {
//            double sigma_inv = 1.0 / (sigmas3d_selected[i] + 1e-6);
//            sigma_inv_sum += sigma_inv;
//            for (int j = 0; j < 3; j++)
//                cws[0][j] += sigma_inv * pws[3 * i + j];
//        }
//
////        std::vector<double> sigmas_start, sigmas_end;
//        std::vector<double> sigmas_lines;
//
//        if (bUseLines)
//        {
////            for (int i = 0; i < number_of_line_correspondences; i++) {
////                const Eigen::Matrix<double,6,6>& S6D = Sigmas3dlines_selected[i];
////                Eigen::Matrix3d SigmaLinesSum = S6D.block<3,3>(0,0) + S6D.block<3,3>(0,3) + S6D.block<3,3>(3,0) + S6D.block<3,3>(3,3);
//////                double sigma_inv_line = 6.0/S6D.trace();
//////                double sigma_inv_end = 6.0/S6D.trace();
//////                sigmas_start.push_back(1.0/sigma_inv_start);
//////                sigmas_end.push_back(1.0/sigma_inv_end);
////                double sigma_line = 0.25*SigmaLinesSum.trace()*1.0/3.0;
////                sigmas_lines.push_back(sigma_line);
////                sigma_inv_sum += sigma_line;
////                for (int j = 0; j < 3; j++)
////                    cws[0][j] += 1.0/sigma_line* 0.5*(Xs_selected[i](j) + Xe_selected[i](j) );
////            }
//        }
//
//        double sigma_sum = 1.0 / (sigma_inv_sum + 1e-6);
//
//        for (int j = 0; j < 3; j++) {
//            cws[0][j] = sigma_sum * cws[0][j];
//        }
//
//        int n = number_of_correspondences;
//        if (bUseLines)
//        {
////            n += number_of_line_correspondences;
//        }
//        // Take C1, C2, and C3 from PCA on the reference points:
//        CvMat *PW0 = cvCreateMat(n, 3, CV_64F);
//
//        double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
//        CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
//        CvMat DC = cvMat(3, 1, CV_64F, dc);
//        CvMat UCt = cvMat(3, 3, CV_64F, uct);
//
//        for (int i = 0; i < number_of_correspondences; i++) {
//            double sigma_remain = sigmas3d_selected[i] - sigma_sum + 1e-6;
//            for (int j = 0; j < 3; j++)
//                PW0->data.db[3 * i + j] = 1.0 / sqrt(fabs(sigma_remain)) * (pws[3 * i + j] - cws[0][j]);
//        }
//        if (bUseLines)
//        {
////            for (int i = 0; i < number_of_line_correspondences; i++) {
////                double sigma_remain_line = sigmas_lines[i] - sigma_sum + 1e-6;
////                for (int j = 0; j < 3; j++) {
////                    PW0->data.db[3*number_of_correspondences + 3 * i + j] = 1.0 / sqrt(fabs(sigma_remain_line)) * ( 0.5*(Xs_selected[i](j) + Xe_selected[i](j)) - cws[0][j]);
////                }
////            }
//        }
//
//        cvMulTransposed(PW0, &PW0tPW0, 1);
//        cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
//
//        cvReleaseMat(&PW0);
//
//        for (int i = 1; i < 4; i++) {
//            double k = sqrt(dc[i - 1] / n);
//            for (int j = 0; j < 3; j++)
//                cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
//        }
//    }

    void PnPsolver::choose_control_points_uncertain_accurate()
    {
        // Take C0 as the reference points centroid:
        cws[0][0] = cws[0][1] = cws[0][2] = 0;

        Eigen::Matrix3d sigma_inv_sum = Eigen::Matrix3d::Zero();
        for (int i = 0; i < number_of_correspondences; i++) {
            Eigen::Matrix3d sigma_inv = (Sigmas3D_selected[i] + 1e-6*Eigen::Matrix3d::Identity()).inverse();
            sigma_inv_sum += sigma_inv;
            Eigen::Vector3d Xp(pws[3*i], pws[3*i+1], pws[3*i+2]);
            Eigen::Vector3d Xpc = sigma_inv*Xp;
            for (int j = 0; j < 3; j++)
                cws[0][j] += Xpc(j);
        }

        Eigen::Matrix3d sigma_sum = (sigma_inv_sum + 1e-6*Eigen::Matrix3d::Identity()).inverse();

        Eigen::Vector3d Xavg(cws[0][0], cws[0][1], cws[0][2]);
        Eigen::Vector3d Xm = sigma_sum * Xavg;
        for (int j = 0; j < 3; j++) {
            cws[0][j] = Xm(j);
        }

        // Take C1, C2, and C3 from PCA on the reference points:
        CvMat *PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

        double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
        CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
        CvMat DC = cvMat(3, 1, CV_64F, dc);
        CvMat UCt = cvMat(3, 3, CV_64F, uct);

        for (int i = 0; i < number_of_correspondences; i++) {
            Eigen::Matrix3d sigma_remain = (Sigmas3D_selected[i] - sigma_sum) + 1e-6*Eigen::Matrix3d::Identity();

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(sigma_remain, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector3d svals = svd.singularValues();
            Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
            S(0,0) = 1.0/sqrt(fabs(svals(0)));
            S(1,1) = 1.0/sqrt(fabs(svals(1)));
            S(2,2) = 1.0/sqrt(fabs(svals(2)));
            Eigen::Matrix3d Sigma3dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

            Eigen::Vector3d Xp(pws[3*i], pws[3*i+1], pws[3*i+2]);
            Eigen::Vector3d Xpc = Sigma3dInv_sqrt * (Xp - Xm);
            for (int j = 0; j < 3; j++)
                PW0->data.db[3 * i + j] = Xpc(j);
        }

        cvMulTransposed(PW0, &PW0tPW0, 1);
        cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

        cvReleaseMat(&PW0);

        for (int i = 1; i < 4; i++) {
            double k = sqrt(dc[i - 1] / number_of_correspondences);
            for (int j = 0; j < 3; j++)
                cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
        }
    }

    void PnPsolver::compute_barycentric_coordinates(void) {
        double cc[3 * 3], cc_inv[3 * 3];
        CvMat CC = cvMat(3, 3, CV_64F, cc);
        CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

        for (int i = 0; i < 3; i++)
            for (int j = 1; j < 4; j++)
                cc[3 * i + j - 1] = cws[j][i] - cws[0][i];

        cvInvert(&CC, &CC_inv, CV_SVD);
        double *ci = cc_inv;
        for (int i = 0; i < number_of_correspondences; i++) {
            double *pi = pws + 3 * i;
            double *a = alphas + 4 * i;

            for (int j = 0; j < 3; j++)
                a[1 + j] =
                        ci[3 * j] * (pi[0] - cws[0][0]) +
                        ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                        ci[3 * j + 2] * (pi[2] - cws[0][2]);
            a[0] = 1.0f - a[1] - a[2] - a[3];
        }

        if (bUseLines)
        {
            std::vector<Eigen::Vector3d> cvecs;
            for (int i = 0; i < 3; i++)
            {
                cvecs.push_back(Eigen::Vector3d(ci[3*i], ci[3*i+1], ci[3*i+2]));
            }
            Eigen::Vector3d cmean(cws[0][0], cws[0][1], cws[0][2]);
            for (int i = 0; i < number_of_line_correspondences; i++) {
                const Eigen::Vector3d& Xs = Xs_selected[i];
                const Eigen::Vector3d& Xe = Xe_selected[i];

                double *a = alphas_start + 4 * i;

                for (int j = 0; j < 3; j++)
                {
                    a[1 + j] = cvecs[j].dot(Xs - cmean);
                }
                a[0] = 1.0f - a[1] - a[2] - a[3];

                a = alphas_end + 4 * i;

                for (int j = 0; j < 3; j++)
                {
                    a[1 + j] = cvecs[j].dot(Xe - cmean);
                }
                a[0] = 1.0f - a[1] - a[2] - a[3];
            }
        }

    }

    void PnPsolver::fill_M(CvMat *M,
                           const int row, const double *as, const double u, const double v) {
        double *M1 = M->data.db + row * 12;
        double *M2 = M1 + 12;

        for (int i = 0; i < 4; i++) {
            M1[3 * i] = as[i] * fu;
            M1[3 * i + 1] = 0.0;
            M1[3 * i + 2] = as[i] * (uc - u);

            M2[3 * i] = 0.0;
            M2[3 * i + 1] = as[i] * fv;
            M2[3 * i + 2] = as[i] * (vc - v);
        }
    }

    void PnPsolver::fill_M_uncertain_line(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, CvMat *M,
                                     const int row, const double *as, const double *ae,
                                     const Eigen::Vector3d& line_eq, const Eigen::Matrix3d& s2d,
                                     const Eigen::Matrix<double,6,6>&S6D, const Eigen::Matrix3d& K,
                                     const Eigen::Vector3d& Xs, const Eigen::Vector3d& Xe, float detSigma2)
    {
//        Eigen::Matrix<double,6,6> S6D;
//        S6D.setZero();
//        S6D = S6D * S6D0.trace() * 1.0/6.0;
//        Eigen::Matrix3d s2d;
//        s2d.setIdentity();
//        for (int i = 0; i < 3; i++)
//        {
//            s2d(i,i) = s2d0(i,i);
//        }
        double * M1 = M->data.db + row * 12;
        double *M2 = M1 + 12;
        Eigen::Vector3d lineEqUn = K.transpose() * line_eq;
        for (int i = 0; i < 4; i++) {
            M1[3 * i] = as[i] * lineEqUn(0);
            M1[3 * i + 1] = as[i] * lineEqUn(1);
            M1[3 * i + 2] = as[i] * lineEqUn(2);

            M2[3 * i] = ae[i] * lineEqUn(0);
            M2[3 * i + 1] = ae[i] * lineEqUn(1);
            M2[3 * i + 2] = ae[i] * lineEqUn(2);
        }

        Eigen::Matrix2d Sigma2;
        Sigma2(0,0) = lineEqUn.transpose() * R_est * S6D.block<3,3>(0,0) * R_est.transpose() * lineEqUn;
        Sigma2(0,1) = lineEqUn.transpose() * R_est * S6D.block<3,3>(0,3) * R_est.transpose() * lineEqUn;
        Sigma2(1,1) = lineEqUn.transpose() * R_est * S6D.block<3,3>(3,3) * R_est.transpose() * lineEqUn;
        Sigma2(1,0) = lineEqUn.transpose() * R_est * S6D.block<3,3>(3,0) * R_est.transpose() * lineEqUn;
        Eigen::Matrix2d Sigma2_2d;
        Eigen::Vector3d Xsc = K*(R_est*Xs+t_est);
        Eigen::Vector3d Xec = K*(R_est*Xe+t_est);

//        //test alphas and Xs
//        std::vector<Eigen::Vector3d> ci;
//        Eigen::Vector3d Xsc2;
//        Xsc2.setZero();
//        for (int j = 0; j < 4; j++)
//        {
//            Eigen::Vector3d cpt(cws[j][0], cws[j][1], cws[j][2]);
//            ci.push_back(R_est*cpt + t_est);
//            Xsc2 += ci[j]*as[j];
//        }
//        Xsc2 = K*Xsc2;
//        std::cout << " test Xs " << (Xsc2-Xsc).norm() << std::endl;

//        Sigma2_2d(0,0) = Xsc.transpose()*s2d*Xsc;
//        Sigma2_2d(1,1) = Xec.transpose()*s2d*Xec;
//        Sigma2_2d(0,1) = Xsc.transpose()*s2d*Xec;
//        Sigma2_2d(1,0) = Xec.transpose()*s2d*Xsc;
//        Eigen::Vector2d n
//        Sigma2_2d =


//        std::cout << "from 3d: " << Sigma2 << std::endl;
//        std::cout << "from 2d: " << Sigma2_2d << std::endl;
        Sigma2_2d = Eigen::Matrix2d ::Identity();
        Sigma2_2d(0,0) = Xsc(2)*Xsc(2);
        Sigma2_2d(1,1) = Xec(2)*Xec(2);
        Sigma2 =  Sigma2_2d * detSigma2 +Sigma2;
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Sigma2+(1e-6)*Eigen::Matrix2d::Identity(), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d svals = svd.singularValues();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
        S(0,0) = 1.0/sqrt(fabs(svals(0)));
        S(1,1) = 1.0/sqrt(fabs(svals(1)));
//        S.setZero();
//        S.setIdentity();
//        S(0,0) = 1.0/Xsc(2)/Xsc(2);
//        S(1,1) = 1.0/Xec(2)/Xec(2);
        Eigen::Matrix2d Sigma2dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

        Eigen::Map<Eigen::Matrix<double, 2, 12, Eigen::RowMajor> > M_pair(M1);
        M_pair = Sigma2dInv_sqrt * M_pair;
    }

    void PnPsolver::fill_M_uncertain_line_simple(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, CvMat *M,
                                          const int row, const double *as, const double *ae,
                                          const Eigen::Vector3d& line_eq, const Eigen::Matrix3d& s2d,
                                          const Eigen::Matrix3d& SLD, const Eigen::Matrix3d& K,
                                          const Eigen::Vector3d& Xs, const Eigen::Vector3d& Xe, float detSigma2)
    {
//        Eigen::Matrix<double,6,6> S6D;
//        S6D.setZero();
//        S6D = S6D * S6D0.trace() * 1.0/6.0;
//        Eigen::Matrix3d s2d;
//        s2d.setIdentity();
//        for (int i = 0; i < 3; i++)
//        {
//            s2d(i,i) = s2d0(i,i);
//        }
        double * M1 = M->data.db + row * 12;
//        double *M2 = M1 + 12;
        Eigen::Vector3d lineEqUn = K.transpose() * line_eq;
        for (int i = 0; i < 4; i++) {
            M1[3 * i] = (ae[i]-as[i]) * lineEqUn(0);
            M1[3 * i + 1] = (ae[i]-as[i]) * lineEqUn(1);
            M1[3 * i + 2] = (ae[i]-as[i]) * lineEqUn(2);
        }

        double sigma = lineEqUn.transpose() * R_est * SLD * R_est.transpose() * lineEqUn  + detSigma2;

//        //test alphas and Xs
//        std::vector<Eigen::Vector3d> ci;
//        Eigen::Vector3d Xsc2;
//        Xsc2.setZero();
//        for (int j = 0; j < 4; j++)
//        {
//            Eigen::Vector3d cpt(cws[j][0], cws[j][1], cws[j][2]);
//            ci.push_back(R_est*cpt + t_est);
//            Xsc2 += ci[j]*as[j];
//        }
//        Xsc2 = K*Xsc2;
//        std::cout << " test Xs " << (Xsc2-Xsc).norm() << std::endl;

//        Sigma2_2d(0,0) = Xsc.transpose()*s2d*Xsc;
//        Sigma2_2d(1,1) = Xec.transpose()*s2d*Xec;
//        Sigma2_2d(0,1) = Xsc.transpose()*s2d*Xec;
//        Sigma2_2d(1,0) = Xec.transpose()*s2d*Xsc;
//        Eigen::Vector2d n
//        Sigma2_2d =


//        std::cout << "from 3d: " << Sigma2 << std::endl;
//        std::cout << "from 2d: " << Sigma2_2d << std::endl;

//        S.setZero();
//        Eigen::Matrix2d Sigma2dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

        Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor> > M_pair(M1);
        M_pair = 1.0/sqrt(sigma)* M_pair;
    }

    void PnPsolver::fill_M_uncertain_line_2(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, CvMat *M,
                                          const int row, const double *as, const double *ae,
                                          const Eigen::Vector3d& line_eq, const Eigen::Matrix3d& s2d,
                                          const Eigen::Matrix<double,6,6>&S6D, const Eigen::Matrix3d& K,
                                          const Eigen::Vector3d& Xs, const Eigen::Vector3d& Xe,
                                          const Eigen::Vector2d& midPt)
    {
//        Eigen::Matrix<double,6,6> S6D;
//        S6D.setIdentity();
//        S6D = S6D * S6D0.trace() * 1.0/6.0;
//        Eigen::Matrix3d s2d;
//        s2d.setIdentity();
//        for (int i = 0; i < 3; i++)
//        {
//            s2d(i,i) = s2d0(i,i);
//        }
        double * M1 = M->data.db + row * 12;
        double *M2 = M1 + 12;
        Eigen::Vector3d lineEqUn = K.transpose() * line_eq;
        for (int i = 0; i < 4; i++) {
            M1[3 * i] = as[i] * lineEqUn(0);
            M1[3 * i + 1] = as[i] * lineEqUn(1);
            M1[3 * i + 2] = as[i] * lineEqUn(2);

            M2[3 * i] = ae[i] * lineEqUn(0);
            M2[3 * i + 1] = ae[i] * lineEqUn(1);
            M2[3 * i + 2] = ae[i] * lineEqUn(2);
        }

        Eigen::Matrix2d Sigma2;
        Sigma2(0,0) = lineEqUn.transpose() * R_est * S6D.block<3,3>(0,0) * R_est.transpose() * lineEqUn;
        Sigma2(0,1) = lineEqUn.transpose() * R_est * S6D.block<3,3>(0,3) * R_est.transpose() * lineEqUn;
        Sigma2(1,1) = lineEqUn.transpose() * R_est * S6D.block<3,3>(3,3) * R_est.transpose() * lineEqUn;
        Sigma2(1,0) = lineEqUn.transpose() * R_est * S6D.block<3,3>(3,0) * R_est.transpose() * lineEqUn;
        Eigen::Matrix2d Sigma2_2d;
        Eigen::Vector3d Xsc = K*(R_est*Xs+t_est);
        Eigen::Vector3d Xec = K*(R_est*Xe+t_est);
        double sigma2dl = midPt.dot(s2d.block<2,2>(0,0) * midPt) + 2*midPt.dot(s2d.block<2,1>(0,2)) + s2d(2,2);
        Sigma2_2d = Eigen::Matrix2d::Identity() * sigma2dl*Xsc(2)*Xsc(2);

//        std::cout << "mpt " << midPt.transpose() << std::endl;
//        std::cout << "Xsc " << Xsc.transpose() << std::endl;
//        std::cout << "Xec " << Xec.transpose() << std::endl;
//        Sigma2_2d(0,0) = Xsc.transpose()*s2d*Xsc;
//        Sigma2_2d(1,1) = Xec.transpose()*s2d*Xec;
//        Sigma2_2d(0,1) = Xsc.transpose()*s2d*Xec;
//        Sigma2_2d(1,0) = Xec.transpose()*s2d*Xsc;
//        Eigen::Vector2d n
//        Sigma2_2d =


//        std::cout << "from 3d: " << Sigma2 << std::endl;
//        std::cout << "from 2d: " << Sigma2_2d << std::endl;
        Sigma2 = Sigma2_2d;
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Sigma2+(1e-6)*Eigen::Matrix2d::Identity(), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d svals = svd.singularValues();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
        S(0,0) = 1.0/sqrt(fabs(svals(0)));
        S(1,1) = 1.0/sqrt(fabs(svals(1)));
        Eigen::Matrix2d Sigma2dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

        Eigen::Map<Eigen::Matrix<double, 2, 12, Eigen::RowMajor> > M_pair(M1);
        M_pair = Sigma2dInv_sqrt * M_pair;
    }

    void PnPsolver::fill_M_line(CvMat *M,
                                          const int row, const double *as, const double *ae,
                                          const Eigen::Vector3d& line_eq,
                                          const Eigen::Matrix3d& K)
    {
        double * M1 = M->data.db + row * 12;
        double *M2 = M1 + 12;
        Eigen::Vector3d lineEqUn = K.transpose() * line_eq;
        for (int i = 0; i < 4; i++) {
            M1[3 * i] = as[i] * lineEqUn(0);
            M1[3 * i + 1] = as[i] * lineEqUn(1);
            M1[3 * i + 2] = as[i] * lineEqUn(2);

            M2[3 * i] = ae[i] * lineEqUn(0);
            M2[3 * i + 1] = ae[i] * lineEqUn(1);
            M2[3 * i + 2] = ae[i] * lineEqUn(2);
        }

    }


    void PnPsolver::fill_M_line_simple(CvMat *M,
                                const int row, const double *as, const double *ae,
                                const Eigen::Vector3d& line_eq,
                                const Eigen::Matrix3d& K)
    {
        double * M1 = M->data.db + row * 12;
        Eigen::Vector3d lineEqUn = K.transpose() * line_eq;
        for (int i = 0; i < 4; i++) {
            M1[3 * i] = (ae[i]-as[i]) * lineEqUn(0);
            M1[3 * i + 1] = (ae[i]-as[i]) * lineEqUn(1);
            M1[3 * i + 2] = (ae[i]-as[i]) * lineEqUn(2);

        }

    }


    void PnPsolver::fill_M_uncertain(const cv::Mat& T_est_cv, CvMat *M,
                                     const int row, const double *as,
                                     const double u, const double v, const double s2d,
                                     const double s3d, double *X)
     {

        Eigen::Vector3d Xp(X[0], X[1], X[2]);
        Eigen::Matrix4d T;
        cv::cv2eigen(T_est_cv, T);
        Eigen::Vector3d Xc = K*(T.block<3, 3>(0, 0) * Xp + T.block<3, 1>(0, 0));
        double sigma_c_2d = Xc(2) * Xc(2) * s2d;

//        Sigma2D_3D(1, 1) = du * du * Sigma3d(3, 3) + 2 * fu * du * Sigma3d(1, 3) + fu * fu * Sigma3d(1, 1);
//        Sigma2D_3D(2, 2) = dv * dv * Sigma3d(3, 3) + 2 * fv * dv * Sigma3d(2, 3) + fv * fv * Sigma3d(2, 2);
//        Sigma2D_3D(1, 2) = du * dv * Sigma3d(3, 3) + fu * dv * Sigma3d(1, 3) + fv * du * Sigma3d(2, 3) + fu * fv * Sigma3d(1, 2);
//        Sigma2D_3D(2, 1) = Sigma2D_3D(1, 2);
//        SigmaUnc = Sigma2D_2D+Sigma2D_3D;
//        SigmaInv = inv(SigmaUnc);%Sigma2D_3D +

        double du = uc -  u;
        double dv = vc - v;
        Eigen::Matrix2d Sigma2d_3d;
        Sigma2d_3d(0, 0) = (du * du + fu*fu) * s3d;
        Sigma2d_3d(1, 1) = (dv * dv + fv*fv) * s3d;
        Sigma2d_3d(0, 1) = (du * dv) * s3d;
        Sigma2d_3d(1, 0) = (du * dv) * s3d;

        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Sigma2d_3d+(1e-6+sigma_c_2d)*Eigen::Matrix2d::Identity(), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d svals = svd.singularValues();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
        S(0,0) = 1.0/sqrt(fabs(svals(0)));
        S(1,1) = 1.0/sqrt(fabs(svals(1)));
        Eigen::Matrix2d Sigma2dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

        double * M1 = M->data.db + row * 12;
        double *M2 = M1 + 12;

        for (int i = 0; i < 4; i++) {
            M1[3 * i] = as[i] * fu;
            M1[3 * i + 1] = 0.0;
            M1[3 * i + 2] = as[i] * du;

            M2[3 * i] = 0.0;
            M2[3 * i + 1] = as[i] * fv;
            M2[3 * i + 2] = as[i] * dv;
        }

        Eigen::Map<Eigen::Matrix<double, 2, 12, Eigen::RowMajor> > M_pair(M1);
        M_pair = Sigma2dInv_sqrt * M_pair;


    }

    void PnPsolver::fill_M_uncertain_accurate(const cv::Mat& T_est_cv, CvMat *M,
                                     const int row, const double *as, const double u, const double v, const double s2d,
                                     const Eigen::Matrix3d& s3dw, double *X)
    {

        Eigen::Vector3d Xp(X[0], X[1], X[2]);
        Eigen::Matrix4d T;
        cv::cv2eigen(T_est_cv, T);
        Eigen::Vector3d Xc = T.block<3, 3>(0, 0) * Xp + T.block<3, 1>(0, 0);
        double sigma_c_2d = Xc(2) * Xc(2) * s2d;

//        Sigma2D_3D(1, 1) = du * du * Sigma3d(3, 3) + 2 * fu * du * Sigma3d(1, 3) + fu * fu * Sigma3d(1, 1);
//        Sigma2D_3D(2, 2) = dv * dv * Sigma3d(3, 3) + 2 * fv * dv * Sigma3d(2, 3) + fv * fv * Sigma3d(2, 2);
//        Sigma2D_3D(1, 2) = du * dv * Sigma3d(3, 3) + fu * dv * Sigma3d(1, 3) + fv * du * Sigma3d(2, 3) + fu * fv * Sigma3d(1, 2);
//        Sigma2D_3D(2, 1) = Sigma2D_3D(1, 2);
//        SigmaUnc = Sigma2D_2D+Sigma2D_3D;
//        SigmaInv = inv(SigmaUnc);%Sigma2D_3D +

        double du = uc -  u;
        double dv = vc - v;
        Eigen::Matrix2d Sigma2d_3d;
        Eigen::Matrix3d s3d = T.block<3, 3>(0, 0)*s3dw*T.block<3, 3>(0, 0).transpose();
        Sigma2d_3d(0, 0) = du * du * s3d(2, 2) + 2 * fu * du * s3d(0, 2) + fu * fu * s3d(0, 0);
        Sigma2d_3d(1, 1) = dv * dv * s3d(2, 2) + 2 * fv * dv * s3d(1, 2) + fv * fv * s3d(1, 1);
        Sigma2d_3d(0, 1) = du * dv * s3d(2, 2) + fu * dv * s3d(0, 2) + fv * du * s3d(1, 2) + fu * fv * s3d(0, 1);
        Sigma2d_3d(1, 0) = Sigma2d_3d(0, 1);

        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Sigma2d_3d+(1e-6+sigma_c_2d )*Eigen::Matrix2d::Identity(), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d svals = svd.singularValues();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
        S(0,0) = 1.0/sqrt(fabs(svals(0)));
        S(1,1) = 1.0/sqrt(fabs(svals(1)));
        Eigen::Matrix2d Sigma2dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();

        double * M1 = M->data.db + row * 12;
        double *M2 = M1 + 12;

        for (int i = 0; i < 4; i++) {
            M1[3 * i] = as[i] * fu;
            M1[3 * i + 1] = 0.0;
            M1[3 * i + 2] = as[i] * du;

            M2[3 * i] = 0.0;
            M2[3 * i + 1] = as[i] * fv;
            M2[3 * i + 2] = as[i] * dv;
        }

        Eigen::Map<Eigen::Matrix<double, 2, 12, Eigen::RowMajor> > M_pair(M1);
        M_pair = Sigma2dInv_sqrt * M_pair;
    }

    void PnPsolver::compute_ccs(const double *betas, const double *ut) {
        for (int i = 0; i < 4; i++)
            ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

        for (int i = 0; i < 4; i++) {
            const double *v = ut + 12 * (11 - i);
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 3; k++)
                    ccs[j][k] += betas[i] * v[3 * j + k];
        }
    }

    void PnPsolver::compute_pcs(void) {
        for (int i = 0; i < number_of_correspondences; i++) {
            double *a = alphas + 4 * i;
            double *pc = pcs + 3 * i;

            for (int j = 0; j < 3; j++)
                pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
        }
    }

    double PnPsolver::compute_pose(double R[3][3], double t[3]) {
        choose_control_points();
        compute_barycentric_coordinates();

        CvMat *M;
        if (bUseLines)//
        {
            M = cvCreateMat(2 * (number_of_correspondences + number_of_line_correspondences), 12, CV_64F);
        } else {
            M = cvCreateMat(2 * (number_of_correspondences), 12, CV_64F);
        }

        for (int i = 0; i < number_of_correspondences; i++)
            fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

        if (bUseLines)//
        {
            Eigen::Matrix3d K;
            K.setIdentity();
            K(0,0) = fu;
            K(1,1) = fv;
            K(0,2) = uc;
            K(1,2) = vc;
            for (int i = 0; i < number_of_line_correspondences; i++)
                fill_M_line(M, 2 * (number_of_correspondences)+2*i, alphas_start + 4 * i, alphas_end + 4*i,
                        lineEqs_selected[i], K);
        }

        double mtm[12 * 12], d[12], ut[12 * 12];
        CvMat MtM = cvMat(12, 12, CV_64F, mtm);
        CvMat D = cvMat(12, 1, CV_64F, d);
        CvMat Ut = cvMat(12, 12, CV_64F, ut);

        cvMulTransposed(M, &MtM, 1);

//        std::cout << MtM << std::endl;

        cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
        cvReleaseMat(&M);

        double l_6x10[6 * 10], rho[6];
        CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
        CvMat Rho = cvMat(6, 1, CV_64F, rho);

        compute_L_6x10(ut, l_6x10);
        compute_rho(rho);

        double Betas[4][4], rep_errors[4];
        double Rs[4][3][3], ts[4][3];

        find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
        gauss_newton(&L_6x10, &Rho, Betas[1]);
        rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

        find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
        gauss_newton(&L_6x10, &Rho, Betas[2]);
        rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

        find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
        gauss_newton(&L_6x10, &Rho, Betas[3]);
        rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

        int N = 1;
        if (rep_errors[2] < rep_errors[1]) N = 2;
        if (rep_errors[3] < rep_errors[N]) N = 3;

        copy_R_and_t(Rs[N], ts[N], R, t);

        return rep_errors[N];
    }

    void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
                                 double R_dst[3][3], double t_dst[3]) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                R_dst[i][j] = R_src[i][j];
            t_dst[i] = t_src[i];
        }
    }

    double PnPsolver::dist2(const double *p1, const double *p2) {
        return
                (p1[0] - p2[0]) * (p1[0] - p2[0]) +
                (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                (p1[2] - p2[2]) * (p1[2] - p2[2]);
    }

    double PnPsolver::dot(const double *v1, const double *v2) {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    double PnPsolver::reprojection_error(const double R[3][3], const double t[3]) {
        double sum2 = 0.0;

        for (int i = 0; i < number_of_correspondences; i++) {
            double *pw = pws + 3 * i;
            double Xc = dot(R[0], pw) + t[0];
            double Yc = dot(R[1], pw) + t[1];
            double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
            double ue = uc + fu * Xc * inv_Zc;
            double ve = vc + fv * Yc * inv_Zc;
            double u = us[2 * i], v = us[2 * i + 1];

            sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
        }

        return sum2 / number_of_correspondences;
    }

    void PnPsolver::estimate_R_and_t(double R[3][3], double t[3]) {
        double pc0[3], pw0[3];

        pc0[0] = pc0[1] = pc0[2] = 0.0;
        pw0[0] = pw0[1] = pw0[2] = 0.0;

        for (int i = 0; i < number_of_correspondences; i++) {
            const double *pc = pcs + 3 * i;
            const double *pw = pws + 3 * i;

            for (int j = 0; j < 3; j++) {
                pc0[j] += pc[j];
                pw0[j] += pw[j];
            }
        }
        for (int j = 0; j < 3; j++) {
            pc0[j] /= number_of_correspondences;
            pw0[j] /= number_of_correspondences;
        }

        double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
        CvMat ABt = cvMat(3, 3, CV_64F, abt);
        CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
        CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
        CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

        cvSetZero(&ABt);
        for (int i = 0; i < number_of_correspondences; i++) {
            double *pc = pcs + 3 * i;
            double *pw = pws + 3 * i;

            for (int j = 0; j < 3; j++) {
                abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
                abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
                abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
            }
        }

        cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

        const double det =
                R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
                R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

        if (det < 0) {
            R[2][0] = -R[2][0];
            R[2][1] = -R[2][1];
            R[2][2] = -R[2][2];
        }

        t[0] = pc0[0] - dot(R[0], pw0);
        t[1] = pc0[1] - dot(R[1], pw0);
        t[2] = pc0[2] - dot(R[2], pw0);
    }

    void PnPsolver::print_pose(const double R[3][3], const double t[3]) {
        cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
        cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
        cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
    }

    void PnPsolver::solve_for_sign(void) {
        if (pcs[2] < 0.0) {
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    ccs[i][j] = -ccs[i][j];

            for (int i = 0; i < number_of_correspondences; i++) {
                pcs[3 * i] = -pcs[3 * i];
                pcs[3 * i + 1] = -pcs[3 * i + 1];
                pcs[3 * i + 2] = -pcs[3 * i + 2];
            }
        }
    }

    double PnPsolver::compute_R_and_t(const double *ut, const double *betas,
                                      double R[3][3], double t[3]) {
        compute_ccs(betas, ut);
        compute_pcs();

        solve_for_sign();

        estimate_R_and_t(R, t);

        return reprojection_error(R, t);
    }

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

    void PnPsolver::find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho,
                                        double *betas) {
        double l_6x4[6 * 4], b4[4];
        CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
        CvMat B4 = cvMat(4, 1, CV_64F, b4);

        for (int i = 0; i < 6; i++) {
            cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
            cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
            cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
            cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
        }

        cvSolve(&L_6x4, Rho, &B4, CV_SVD);

        if (b4[0] < 0) {
            betas[0] = sqrt(-b4[0]);
            betas[1] = -b4[1] / betas[0];
            betas[2] = -b4[2] / betas[0];
            betas[3] = -b4[3] / betas[0];
        } else {
            betas[0] = sqrt(b4[0]);
            betas[1] = b4[1] / betas[0];
            betas[2] = b4[2] / betas[0];
            betas[3] = b4[3] / betas[0];
        }
    }

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

    void PnPsolver::find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho,
                                        double *betas) {
        double l_6x3[6 * 3], b3[3];
        CvMat L_6x3 = cvMat(6, 3, CV_64F, l_6x3);
        CvMat B3 = cvMat(3, 1, CV_64F, b3);

        for (int i = 0; i < 6; i++) {
            cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
            cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
            cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
        }

        cvSolve(&L_6x3, Rho, &B3, CV_SVD);

        if (b3[0] < 0) {
            betas[0] = sqrt(-b3[0]);
            betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
        } else {
            betas[0] = sqrt(b3[0]);
            betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
        }

        if (b3[1] < 0) betas[0] = -betas[0];

        betas[2] = 0.0;
        betas[3] = 0.0;
    }

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

    void PnPsolver::find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho,
                                        double *betas) {
        double l_6x5[6 * 5], b5[5];
        CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
        CvMat B5 = cvMat(5, 1, CV_64F, b5);

        for (int i = 0; i < 6; i++) {
            cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
            cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
            cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
            cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
            cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
        }

        cvSolve(&L_6x5, Rho, &B5, CV_SVD);

        if (b5[0] < 0) {
            betas[0] = sqrt(-b5[0]);
            betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
        } else {
            betas[0] = sqrt(b5[0]);
            betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
        }
        if (b5[1] < 0) betas[0] = -betas[0];
        betas[2] = b5[3] / betas[0];
        betas[3] = 0.0;
    }

    void PnPsolver::compute_L_6x10(const double *ut, double *l_6x10) {
        const double *v[4];

        v[0] = ut + 12 * 11;
        v[1] = ut + 12 * 10;
        v[2] = ut + 12 * 9;
        v[3] = ut + 12 * 8;

        double dv[4][6][3];

        for (int i = 0; i < 4; i++) {
            int a = 0, b = 1;
            for (int j = 0; j < 6; j++) {
                dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
                dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
                dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

                b++;
                if (b > 3) {
                    a++;
                    b = a + 1;
                }
            }
        }

        for (int i = 0; i < 6; i++) {
            double *row = l_6x10 + 10 * i;

            row[0] = dot(dv[0][i], dv[0][i]);
            row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
            row[2] = dot(dv[1][i], dv[1][i]);
            row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
            row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
            row[5] = dot(dv[2][i], dv[2][i]);
            row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
            row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
            row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
            row[9] = dot(dv[3][i], dv[3][i]);
        }
    }

    void PnPsolver::compute_rho(double *rho) {
        rho[0] = dist2(cws[0], cws[1]);
        rho[1] = dist2(cws[0], cws[2]);
        rho[2] = dist2(cws[0], cws[3]);
        rho[3] = dist2(cws[1], cws[2]);
        rho[4] = dist2(cws[1], cws[3]);
        rho[5] = dist2(cws[2], cws[3]);
    }

    void PnPsolver::compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                                 double betas[4], CvMat *A, CvMat *b) {
        for (int i = 0; i < 6; i++) {
            const double *rowL = l_6x10 + i * 10;
            double *rowA = A->data.db + i * 4;

            rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
            rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
            rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
            rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

            cvmSet(b, i, 0, rho[i] -
                            (
                                    rowL[0] * betas[0] * betas[0] +
                                    rowL[1] * betas[0] * betas[1] +
                                    rowL[2] * betas[1] * betas[1] +
                                    rowL[3] * betas[0] * betas[2] +
                                    rowL[4] * betas[1] * betas[2] +
                                    rowL[5] * betas[2] * betas[2] +
                                    rowL[6] * betas[0] * betas[3] +
                                    rowL[7] * betas[1] * betas[3] +
                                    rowL[8] * betas[2] * betas[3] +
                                    rowL[9] * betas[3] * betas[3]
                            ));
        }
    }

    void PnPsolver::gauss_newton(const CvMat *L_6x10, const CvMat *Rho,
                                 double betas[4]) {
        const int iterations_number = 5;

        double a[6 * 4], b[6], x[4];
        CvMat A = cvMat(6, 4, CV_64F, a);
        CvMat B = cvMat(6, 1, CV_64F, b);
        CvMat X = cvMat(4, 1, CV_64F, x);

        for (int k = 0; k < iterations_number; k++) {
            compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db,
                                         betas, &A, &B);
            qr_solve(&A, &B, &X);

            for (int i = 0; i < 4; i++)
                betas[i] += x[i];
        }
    }

    void PnPsolver::qr_solve(CvMat *A, CvMat *b, CvMat *X) {
        static int max_nr = 0;
        static double *A1, *A2;

        const int nr = A->rows;
        const int nc = A->cols;

        if (max_nr != 0 && max_nr < nr) {
            delete[] A1;
            delete[] A2;
        }
        if (max_nr < nr) {
            max_nr = nr;
            A1 = new double[nr];
            A2 = new double[nr];
        }

        double *pA = A->data.db, *ppAkk = pA;
        for (int k = 0; k < nc; k++) {
            double *ppAik = ppAkk, eta = fabs(*ppAik);
            for (int i = k + 1; i < nr; i++) {
                double elt = fabs(*ppAik);
                if (eta < elt) eta = elt;
                ppAik += nc;
            }

            if (eta == 0) {
                A1[k] = A2[k] = 0.0;
                cerr << "God damnit, A is singular, this shouldn't happen." << endl;
                return;
            } else {
                double *ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
                for (int i = k; i < nr; i++) {
                    *ppAik *= inv_eta;
                    sum += *ppAik * *ppAik;
                    ppAik += nc;
                }
                double sigma = sqrt(sum);
                if (*ppAkk < 0)
                    sigma = -sigma;
                *ppAkk += sigma;
                A1[k] = sigma * *ppAkk;
                A2[k] = -eta * sigma;
                for (int j = k + 1; j < nc; j++) {
                    double *ppAik = ppAkk, sum = 0;
                    for (int i = k; i < nr; i++) {
                        sum += *ppAik * ppAik[j - k];
                        ppAik += nc;
                    }
                    double tau = sum / A1[k];
                    ppAik = ppAkk;
                    for (int i = k; i < nr; i++) {
                        ppAik[j - k] -= tau * *ppAik;
                        ppAik += nc;
                    }
                }
            }
            ppAkk += nc + 1;
        }

        // b <- Qt b
        double *ppAjj = pA, *pb = b->data.db;
        for (int j = 0; j < nc; j++) {
            double *ppAij = ppAjj, tau = 0;
            for (int i = j; i < nr; i++) {
                tau += *ppAij * pb[i];
                ppAij += nc;
            }
            tau /= A1[j];
            ppAij = ppAjj;
            for (int i = j; i < nr; i++) {
                pb[i] -= tau * *ppAij;
                ppAij += nc;
            }
            ppAjj += nc + 1;
        }

        // X = R-1 b
        double *pX = X->data.db;
        pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
        for (int i = nc - 2; i >= 0; i--) {
            double *ppAij = pA + i * nc + (i + 1), sum = 0;

            for (int j = i + 1; j < nc; j++) {
                sum += *ppAij * pX[j];
                ppAij++;
            }
            pX[i] = (pb[i] - sum) / A2[i];
        }
    }


    void PnPsolver::relative_error(double &rot_err, double &transl_err,
                                   const double Rtrue[3][3], const double ttrue[3],
                                   const double Rest[3][3], const double test[3]) {
        double qtrue[4], qest[4];

        mat_to_quat(Rtrue, qtrue);
        mat_to_quat(Rest, qest);

        double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
                               (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
                               (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
                               (qtrue[3] - qest[3]) * (qtrue[3] - qest[3])) /
                          sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

        double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
                               (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
                               (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
                               (qtrue[3] + qest[3]) * (qtrue[3] + qest[3])) /
                          sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

        rot_err = min(rot_err1, rot_err2);

        transl_err =
                sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
                     (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
                     (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
                sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
    }

    void PnPsolver::mat_to_quat(const double R[3][3], double q[4]) {
        double tr = R[0][0] + R[1][1] + R[2][2];
        double n4;

        if (tr > 0.0f) {
            q[0] = R[1][2] - R[2][1];
            q[1] = R[2][0] - R[0][2];
            q[2] = R[0][1] - R[1][0];
            q[3] = tr + 1.0f;
            n4 = q[3];
        } else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
            q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
            q[1] = R[1][0] + R[0][1];
            q[2] = R[2][0] + R[0][2];
            q[3] = R[1][2] - R[2][1];
            n4 = q[0];
        } else if (R[1][1] > R[2][2]) {
            q[0] = R[1][0] + R[0][1];
            q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
            q[2] = R[2][1] + R[1][2];
            q[3] = R[2][0] - R[0][2];
            n4 = q[1];
        } else {
            q[0] = R[2][0] + R[0][2];
            q[1] = R[2][1] + R[1][2];
            q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
            q[3] = R[0][1] - R[1][0];
            n4 = q[2];
        }
        double scale = 0.5f / double(sqrt(n4));

        q[0] *= scale;
        q[1] *= scale;
        q[2] *= scale;
        q[3] *= scale;
    }

    int FindBestSolutionReproj(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                               const Eigen::Matrix3d& K,
                                     const posevector& poses, double* min_err_p)
    {
        double min_err = 1e100;
        int sol_ind = -1;
        for (int i = 0; i < poses.size(); i++)
        {
            auto T = poses[i];
            double err = 0;
            double cnt = 0.0;
            bool is_behind = false;
            for (int j = 0; j < XX.size(); j++)
            {
                Eigen::Vector3d Xp = XX[j];
                Eigen::Vector3d xc = K*(T.block<3,3>(0,0) * Xp + T.block<3,1>(0,3));
                if (xc(2) < 0)
                {
                    is_behind = true;
                }
                xc = xc / xc(2);
                err += (xc.segment<2>(0) - xx[j]).norm();
                cnt += 1;
            }
            err = err/cnt;
            if (err < min_err && !is_behind)
            {
                min_err = err;
                sol_ind = i;
            }
        }
        *min_err_p = min_err;
        return sol_ind;
    }

    void PnPsolver::compute_pose_dlsu(const cv::Mat& T_est_cv, double R[3][3], double t[3], bool is_two_stage)
    {
//        void DLSU(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
//                  const std::vector<float>& sigmas3d,
//                  const std::vector<float>& sigmas2d_norm,
//                  const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p);
        std::vector<Eigen::Vector3d> XX;
        vec2d xx;
        std::vector<float> sigmas2d_norm;
        Eigen::Vector3d Xm;
        Xm.setZero();
        for (int i = 0; i < number_of_correspondences; i++)
        {
            Eigen::Vector3d X(pws[3*i], pws[3*i+1], pws[3*i+2]);
            XX.push_back(X);
            Xm = Xm + X;

            Eigen::Vector2d x(us[2*i], us[2*i+1]);
            xx.push_back(x);

            sigmas2d_norm.push_back(sigmas2d_selected[i]/fu/fu);
        }

        Xm = Xm / float(number_of_correspondences);
        for (int i = 0; i < number_of_correspondences; i++)
        {
            XX[i] = XX[i] - Xm;
        }



        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0,0) = fu;
        K(1,1) = fv;
        K(0,2) = uc;
        K(1,2) = vc;
        Eigen::Matrix4d T_est;
        cv::cv2eigen(T_est_cv, T_est);
        posevector sols;

        mat6dvector Sigmas3DLinesBalls;
        std::vector<Eigen::Matrix3d> Sigmas2DLinesBalls;

        if (bUseLines)
        {

            for (size_t i = 0; i < Xs_selected.size(); i++)
            {
                Eigen::Matrix<double,6,6> S6D;
                S6D.setIdentity();
                S6D = S6D * Sigmas3dlines_selected[i].trace() * 1.0/6.0;
                Eigen::Matrix3d s2d;
                s2d.setIdentity();
                for (int j = 0; j < 3; j++)
                {
                    s2d(j,j) = Sigmas2dlinesN_selected[i](j,j);
                }
                Sigmas3DLinesBalls.push_back(S6D);
                Sigmas2DLinesBalls.push_back(s2d);
            }
//            void DLSULines(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
//                           const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
//                           const std::vector<Eigen::Vector3d>& l2ds,
//                           const std::vector<float>& sigmas3d,
//                           const std::vector<float>& sigmas2d_norm,
//                           const mat6dvector& sigmas3d_lines,
//                           const std::vector<Eigen::Matrix3d>& sigmas2d_lines,
//                           const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p);

            DLSULines(XX, xx, Xs_selected, Xe_selected, lineEqsN_selected, sigmas3d_selected, sigmas2d_norm,
                      Sigmas3dlines_selected, Sigmas2dlinesN_selected,
                      sigmasDetLineSelected,
                      K, T_est.block<3,3>(0,0), T_est.block<3,1>(0,3), &sols);
        } else {
            DLSU(XX, xx, sigmas3d_selected, sigmas2d_norm, K, T_est.block<3,3>(0,0), T_est.block<3,1>(0,3), &sols);
        }

//        sols.clear();
        if (sols.size() == 0)
        {
//            std::cout << " cycle start " << std::endl;
            int cnt = 0;
            while (cnt < 5 && sols.size() == 0) {
                cv::Mat rvec(3, 1, CV_64FC1);
                cv::theRNG().fill(rvec, cv::RNG::NORMAL, 0, 6.28);
                cv::Mat Rr;
                cv::Rodrigues(rvec, Rr);
                Eigen::Matrix3d Rr_eig;
                cv::cv2eigen(Rr, Rr_eig);
//                std::cout << " random rot " << Rr_eig << std::endl;
                std::vector<Eigen::Vector3d> XXr(XX.size());

                mat6dvector Sigmas3dlines_selected_rot;
                for (int i = 0; i < XX.size(); i++) {
                    XXr[i] = Rr_eig * XX[i];
                }
                std::vector<Eigen::Vector3d> Xsr(Xs_selected.size());
                std::vector<Eigen::Vector3d> Xer(Xs_selected.size());
                if (bUseLines)
                {
                    for (int i = 0; i < Xs_selected.size(); i++)
                    {
                        Xsr.push_back(Rr_eig * Xs_selected[i]);
                        Xer.push_back(Rr_eig * Xe_selected[i]);
                        Eigen::Matrix<double,6,6> SRot;
                        for (int ri = 0; ri < 2; ri++)
                        {
                            for (int ci = 0; ci < 2; ci++)
                            {
                                SRot.block<3,3>(3*ri,3*ci) = Rr_eig * Sigmas3dlines_selected[i].block<3,3>(3*ri,3*ci) * Rr_eig.transpose();
                            }
                        }

                        Sigmas3dlines_selected_rot.push_back(SRot);
                    }
                }
                Eigen::Matrix3d Rr_est = T_est.block<3, 3>(0, 0) * Rr_eig.transpose();
                if (bUseLines) {
                    DLSULines(XXr, xx, Xsr, Xer, lineEqsN_selected, sigmas3d_selected, sigmas2d_norm,
                              Sigmas3dlines_selected_rot, Sigmas2dlinesN_selected,
                              sigmasDetLineSelected,
                              K, Rr_est, T_est.block<3,1>(0,3), &sols);
                } else {
                    DLSU(XXr, xx, sigmas3d_selected, sigmas2d_norm, K, Rr_est, T_est.block<3, 1>(0, 3), &sols);
                }
                if (sols.size()>0)
                {
                    for (int i = 0; i < sols.size(); i++)
                    {
                        sols[i].block<3,3>(0,0) = sols[i].block<3,3>(0,0) * Rr_eig;
                    }
                }
                cnt++;
            }
        }

        if (is_two_stage)
        {
            if (sols.size()>0)
            {
                Eigen::Matrix4d T_est = sols[0];
                posevector sols_fine;

                if (bUseLines)
                {
                    DLSULines_accurate(XX, xx, Xs_selected, Xe_selected, lineEqsN_selected, Sigmas3D_selected, sigmas2d_selected,
                                       Sigmas3dlines_selected, Sigmas2dlinesN_selected,
                                       sigmasDetLineSelected,
                                       K, T_est.block<3, 3>(0, 0),T_est.block<3, 1>(0, 3), &sols_fine);

                } else {
                    DLSU_accurate(XX, xx, Sigmas3D_selected, sigmas2d_selected, K, T_est.block<3, 3>(0, 0),
                                  T_est.block<3, 1>(0, 3), &sols_fine);
                }
                for (int i = 0; i < sols_fine.size(); i++)
                {
                    sols.push_back(sols_fine[i]);
                }
            }
        }

        sols.push_back(T_est);

        double min_err;
        int sol_ind = FindBestSolutionReproj(XX, xx, K, sols, &min_err);

        if (sol_ind>=0)
        {
            Eigen::Matrix4d T = sols[sol_ind];
            Eigen::Vector3d shift_rot = T.block<3,3>(0,0) * Xm;
            for (int ri = 0; ri < 3; ri++)
            {
                for (int ci = 0; ci < 3; ci++) {
                    mRi[ri][ci] = T(ri, ci);
                }
                mti[ri] = T(ri, 3) - shift_rot(ri);
            }
        }
    }

    void PnPsolver::compute_pose_dlsu_accurate(const cv::Mat& T_est_cv, double R[3][3], double t[3])
    {
//        void DLSU(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
//                  const std::vector<float>& sigmas3d,
//                  const std::vector<float>& sigmas2d_norm,
//                  const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p);
        std::vector<Eigen::Vector3d> XX;
        vec2d xx;
        std::vector<float> sigmas2d_norm;
        for (int i = 0; i < number_of_correspondences; i++)
        {
            Eigen::Vector3d X(pws[3*i], pws[3*i+1], pws[3*i+2]);
            XX.push_back(X);

            Eigen::Vector2d x(us[2*i], us[2*i+1]);
            xx.push_back(x);

            sigmas2d_norm.push_back(sigmas2d_selected[i]/fu/fu);
        }

        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0,0) = fu;
        K(1,1) = fv;
        K(0,2) = uc;
        K(1,2) = vc;
        Eigen::Matrix4d T_est;
        cv::cv2eigen(T_est_cv, T_est);
        posevector sols;
        DLSU_accurate(XX, xx, Sigmas3D_selected, sigmas2d_norm, K, T_est.block<3,3>(0,0), T_est.block<3,1>(0,3), &sols);

        double min_err;
        int sol_ind = FindBestSolutionReproj(XX, xx, K, sols, &min_err);

        if (sol_ind>=0)
        {
            Eigen::Matrix4d T = sols[sol_ind];
            for (int ri = 0; ri < 3; ri++)
            {
                for (int ci = 0; ci < 3; ci++) {
                    mRi[ri][ci] = T(ri, ci);
                }
                mti[ri] = T(ri, 3);
            }
        }
    }


    double PnPsolver::compute_pose_uncertain(const cv::Mat& T_est_cv, double R[3][3], double t[3]) {
        choose_control_points_uncertain();
        compute_barycentric_coordinates();

        CvMat *M;

        if (bUseLines)//bUseLines
        {
            M = cvCreateMat(2 * (number_of_correspondences + number_of_line_correspondences), 12, CV_64F);
        } else {
            M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);
        }

        for (int i = 0; i < number_of_correspondences; i++)
            fill_M_uncertain(T_est_cv, M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1], sigmas2d_selected[i],
                             sigmas3d_selected[i], pws + 3 * i);

        if (bUseLines) {//bUseLines
            Eigen::Matrix4d T_est;
            cv::cv2eigen(T_est_cv, T_est);
            Eigen::Matrix3d R_est = T_est.block<3,3>(0,0);
            Eigen::Vector3d t_est = T_est.block<3,1>(0,3);
            Eigen::Matrix3d K;
            K.setIdentity();
            K(0,0) = fu;
            K(1,1) = fv;
            K(0,2) = uc;
            K(1,2) = vc;
            for (int i = 0; i < number_of_line_correspondences; i++) {
//                fill_M_uncertain_line_simple(R_est, t_est, M,
//                                       2*number_of_correspondences+i, alphas_start + 4*i, alphas_end + 4*i,
//                                       lineEqs_selected[i], Sigmas2dlines_selected[i],
//                                       Sigmas_LD_selected[i],
//                                       K,
//                                       Xs_selected[i], Xe_selected[i],
//                                      sigmasDetLineSelected[i]);
                fill_M_uncertain_line(R_est, t_est, M,
                                       2*(number_of_correspondences+i), alphas_start + 4*i, alphas_end + 4*i,
                                       lineEqs_selected[i], Sigmas2dlines_selected[i],
                                       Sigmas3dlines_selected[i],
                                       K,
                                       Xs_selected[i], Xe_selected[i],
                                      sigmasDetLineSelected[i]);

//                fill_M_uncertain_line_2(R_est, t_est, M,
//                                      2*(number_of_correspondences+i), alphas_start + 4*i, alphas_end + 4*i,
//                                      lineEqs_selected[i], Sigmas2dlines_selected[i],
//                                      Sigmas3dlines_selected[i], K,
//                                      Xs_selected[i], Xe_selected[i],
//                                        0.5*(start_pts_2d_selected[i]+end_pts_2d_selected[i]));
            }
        }

        double mtm[12 * 12], d[12], ut[12 * 12];
        CvMat MtM = cvMat(12, 12, CV_64F, mtm);
        CvMat D = cvMat(12, 1, CV_64F, d);
        CvMat Ut = cvMat(12, 12, CV_64F, ut);

        cvMulTransposed(M, &MtM, 1);
        cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
        cvReleaseMat(&M);

        double l_6x10[6 * 10], rho[6];
        CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
        CvMat Rho = cvMat(6, 1, CV_64F, rho);

        compute_L_6x10(ut, l_6x10);
        compute_rho(rho);

        double Betas[4][4], rep_errors[4];
        double Rs[4][3][3], ts[4][3];

        find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
        gauss_newton(&L_6x10, &Rho, Betas[1]);
        rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

        find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
        gauss_newton(&L_6x10, &Rho, Betas[2]);
        rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

        find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
        gauss_newton(&L_6x10, &Rho, Betas[3]);
        rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

        int N = 1;
        if (rep_errors[2] < rep_errors[1]) N = 2;
        if (rep_errors[3] < rep_errors[N]) N = 3;

        copy_R_and_t(Rs[N], ts[N], R, t);

        return rep_errors[N];
    }

    double PnPsolver::compute_pose_uncertain_accurate(const cv::Mat& T_est_cv, double R[3][3], double t[3])
    {
        choose_control_points_uncertain_accurate();
        compute_barycentric_coordinates();

        CvMat *M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);

        for (int i = 0; i < number_of_correspondences; i++)
            fill_M_uncertain_accurate(T_est_cv, M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1], sigmas2d_selected[i],
                             Sigmas3D_selected[i], pws + 3 * i);

        double mtm[12 * 12], d[12], ut[12 * 12];
        CvMat MtM = cvMat(12, 12, CV_64F, mtm);
        CvMat D = cvMat(12, 1, CV_64F, d);
        CvMat Ut = cvMat(12, 12, CV_64F, ut);

        cvMulTransposed(M, &MtM, 1);
        cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
        cvReleaseMat(&M);

        double l_6x10[6 * 10], rho[6];
        CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
        CvMat Rho = cvMat(6, 1, CV_64F, rho);

        compute_L_6x10(ut, l_6x10);
        compute_rho(rho);

        double Betas[4][4], rep_errors[4];
        double Rs[4][3][3], ts[4][3];

        find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
        gauss_newton(&L_6x10, &Rho, Betas[1]);
        rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

        find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
        gauss_newton(&L_6x10, &Rho, Betas[2]);
        rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

        find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
        gauss_newton(&L_6x10, &Rho, Betas[3]);
        rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

        int N = 1;
        if (rep_errors[2] < rep_errors[1]) N = 2;
        if (rep_errors[3] < rep_errors[N]) N = 3;

        copy_R_and_t(Rs[N], ts[N], R, t);

        return rep_errors[N];
    }

    bool PnPsolver::GetLineInliers(vector<bool> &vbLineInliers, int &nLineInliers)
    {
        vbLineInliers = mvbInliersLines;
        nLineInliers = mnInliersLinesi;
        return bUseLines;
    }


} //namespace ORB_SLAM


