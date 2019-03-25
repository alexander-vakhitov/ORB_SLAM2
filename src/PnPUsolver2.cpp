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

#include <iostream>

#include "PnPUsolver2.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace std;

namespace ORB_SLAM2
{


PnPUsolver2::PnPUsolver2(const std::vector<Eigen::Matrix3d>& sigmas3d, const std::vector<cv::Point2f>& p2D,
                         const std::vector<float>& sigma2, const std::vector<cv::Point3f>& p3D,
                         const std::vector<size_t>& keyPointIndices, const std::vector<size_t>& allIndices,
                         double fu, double fv, double uc, double vc, int nMapPoints):
    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0),
    Sigmas3d(sigmas3d),
    mvP2D(p2D), mvSigma2(sigma2), mvP3Dw(p3D), mvKeyPointIndices(keyPointIndices), mvAllIndices(allIndices),
    fu(fu), fv(fv), uc(uc), vc(vc), nMapPoints(nMapPoints)
{

    SetRansacParameters();
}

PnPUsolver2::~PnPUsolver2()
{
  delete [] pws;
  delete [] us;
  delete [] alphas;
  delete [] pcs;
}


void PnPUsolver2::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    N = mvP2D.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    int nMinInliers = N*mRansacEpsilon;
    if(nMinInliers<mRansacMinInliers)
        nMinInliers=mRansacMinInliers;
    if(nMinInliers<minSet)
        nMinInliers=minSet;
    mRansacMinInliers = nMinInliers;

    if(mRansacEpsilon<(float)mRansacMinInliers/N)
        mRansacEpsilon=(float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(mRansacEpsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mvMaxError.resize(mvSigma2.size());
    for(size_t i=0; i<mvSigma2.size(); i++)
        mvMaxError[i] = mvSigma2[i]*th2;
}

cv::Mat PnPUsolver2::find(vector<bool> &vbInliers, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers,nInliers);    
}

cv::Mat PnPUsolver2::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers=0;

    set_maximum_number_of_correspondences(mRansacMinSet);

    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();

        vAvailableIndices = mvAllIndices;

        mvChosenIndices.resize(mRansacMinSet);

        // Get min set of points
        for(short i = 0; i < mRansacMinSet; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();

            mvChosenIndices[i] = idx;
        }

        // Compute camera pose
        compute_pose(mRi, mti, true);

        // Check inliers
        CheckInliers();

        if(mnInliersi>=mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            if(mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3,3,CV_64F,mRi);
                cv::Mat tcw(3,1,CV_64F,mti);
                Rcw.convertTo(Rcw,CV_32F);
                tcw.convertTo(tcw,CV_32F);
                mBestTcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
            }

            Eigen::Matrix4d T_est;
            cv::cv2eigen(mBestTcw, T_est);

            if(Refine(T_est.block<3,3>(0,0), T_est.block<3,1>(0,3)))
            {
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(nMapPoints,false);
                for(int i=0; i<N; i++)
                {
                    if(mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mRefinedTcw.clone();
            }

        }
    }

    if(mnIterations>=mRansacMaxIts)
    {
        bNoMore=true;
        if(mnBestInliers>=mRansacMinInliers)
        {
            nInliers=mnBestInliers;
            vbInliers = vector<bool>(nMapPoints,false);
            for(int i=0; i<N; i++)
            {
                if(mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }

    return cv::Mat();
}

bool PnPUsolver2::Refine(const Eigen::Matrix3d& Rest, const Eigen::Vector3d& test)
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for(size_t i=0; i<mvbBestInliers.size(); i++)
    {
        if(mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    set_maximum_number_of_correspondences(vIndices.size());

    reset_correspondences();

    mvChosenIndices.resize(vIndices.size());

    for(size_t i=0; i<vIndices.size(); i++)
    {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
        mvChosenIndices[i] = idx;
    }

    // Compute camera pose
    compute_pose(mRi, mti, false, Rest, test);

    // Check inliers
    CheckInliers();

    mnRefinedInliers =mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if(mnInliersi>mRansacMinInliers)
    {
        cv::Mat Rcw(3,3,CV_64F,mRi);
        cv::Mat tcw(3,1,CV_64F,mti);
        Rcw.convertTo(Rcw,CV_32F);
        tcw.convertTo(tcw,CV_32F);
        mRefinedTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
        return true;
    }

    return false;
}


void PnPUsolver2::CheckInliers()
{
    mnInliersi=0;

    for(int i=0; i<N; i++)
    {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];

        float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
        float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
        float invZc = 1/(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);

        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;

        float distX = P2D.x-ue;
        float distY = P2D.y-ve;

        float error2 = distX*distX+distY*distY;

        if(error2<mvMaxError[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
        {
            mvbInliersi[i]=false;
        }
    }
}


void PnPUsolver2::set_maximum_number_of_correspondences(int n)
{
  if (maximum_number_of_correspondences < n) {
    if (pws != 0) delete [] pws;
    if (us != 0) delete [] us;
    if (alphas != 0) delete [] alphas;
    if (pcs != 0) delete [] pcs;

    maximum_number_of_correspondences = n;
    pws = new double[3 * maximum_number_of_correspondences];
    us = new double[2 * maximum_number_of_correspondences];
    alphas = new double[4 * maximum_number_of_correspondences];
    pcs = new double[3 * maximum_number_of_correspondences];
  }
}

void PnPUsolver2::reset_correspondences(void)
{
  number_of_correspondences = 0;
}

void PnPUsolver2::add_correspondence(double X, double Y, double Z, double u, double v)
{
  pws[3 * number_of_correspondences    ] = X;
  pws[3 * number_of_correspondences + 1] = Y;
  pws[3 * number_of_correspondences + 2] = Z;

  us[2 * number_of_correspondences    ] = u;
  us[2 * number_of_correspondences + 1] = v;

  number_of_correspondences++;
}

void PnPUsolver2::choose_control_points(bool is_minimal)
{
    Eigen::Matrix3d S;
    S.setZero();
    Eigen::Vector3d xm;

    if (is_minimal)
    {
//         Take C0 as the reference points centroid:
      cws[0][0] = cws[0][1] = cws[0][2] = 0;
      for(int i = 0; i < number_of_correspondences; i++)
        for(int j = 0; j < 3; j++)
          cws[0][j] += pws[3 * i + j];

      for(int j = 0; j < 3; j++)
        cws[0][j] /= number_of_correspondences;

    } else {
        xm.setZero();
        for (int i = 0; i < number_of_correspondences; i++)
        {
            int ind = mvChosenIndices[i];
            Eigen::Vector3d X(mvP3Dw[ind].x, mvP3Dw[ind].y, mvP3Dw[ind].z);
            Eigen::Matrix3d SigmaI = (Sigmas3d[ind]+ Eigen::Matrix3d::Identity() * 1e-6).inverse();//
            xm += SigmaI * X;
            S += SigmaI;
        }
        xm = S.inverse() * xm;
        for (int j = 0; j < 3; j++)
        {
            cws[0][j] = xm(j);
        }
    }

  // Take C1, C2, and C3 from PCA on the reference points:
  CvMat * PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

  double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
  CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
  CvMat DC      = cvMat(3, 1, CV_64F, dc);
  CvMat UCt     = cvMat(3, 3, CV_64F, uct);
    if (!is_minimal)
    {
        std::vector<Eigen::Matrix3d> Sigmas3dComp;
        for (int i = 0; i < number_of_correspondences; i++)
        {
            int ind = mvChosenIndices[i];
            Sigmas3dComp.push_back(Sigmas3d[ind] + Eigen::Matrix3d::Identity() * 1e-9 - S.inverse());
        }
        for (int i = 0; i < number_of_correspondences; i++)
        {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(Sigmas3dComp[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector3d svals = svd.singularValues();
            Eigen::Matrix3d Si_diag = Eigen::Matrix3d::Identity();
            Si_diag(0, 0) = 1.0 / sqrt(svals(0));
            Si_diag(1, 1) = 1.0 / sqrt(svals(1));
            Si_diag(2, 2) = 1.0 / sqrt(svals(2));
            Eigen::Matrix3d Si = svd.matrixU() * Si_diag * svd.matrixU().transpose();
            Eigen::Vector3d p(pws[3 * i], pws[3 * i + 1], pws[3 * i + 2]);
            Eigen::Vector3d dp_norm = Si*(p - xm);
            for (int j = 0; j < 3; j++) {
                PW0->data.db[3 * i + j] = dp_norm[j];
            }
        }

    } else {
    for(int i = 0; i < number_of_correspondences; i++)
        for(int j = 0; j < 3; j++)
            PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];
    }

  cvMulTransposed(PW0, &PW0tPW0, 1);
  cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

  cvReleaseMat(&PW0);

  for(int i = 1; i < 4; i++) {
    double k = sqrt(dc[i - 1] / number_of_correspondences);
    for(int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
  }
}

void PnPUsolver2::compute_barycentric_coordinates(void)
{
  double cc[3 * 3], cc_inv[3 * 3];
  CvMat CC     = cvMat(3, 3, CV_64F, cc);
  CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

  for(int i = 0; i < 3; i++)
    for(int j = 1; j < 4; j++)
      cc[3 * i + j - 1] = cws[j][i] - cws[0][i];

  cvInvert(&CC, &CC_inv, CV_SVD);
  double * ci = cc_inv;
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pi = pws + 3 * i;
    double * a = alphas + 4 * i;

    for(int j = 0; j < 3; j++)
      a[1 + j] =
	ci[3 * j    ] * (pi[0] - cws[0][0]) +
	ci[3 * j + 1] * (pi[1] - cws[0][1]) +
	ci[3 * j + 2] * (pi[2] - cws[0][2]);
    a[0] = 1.0f - a[1] - a[2] - a[3];
  }
}

void PnPUsolver2::fill_M(CvMat * M,
		  const int row, const double * as, const double u, const double v, const Eigen::Matrix3d& Sigma3d,
		  bool is_weighted, double de, double sigma2d)
{
  double * M1 = M->data.db + row * 12;
  double * M2 = M1 + 12;

  for(int i = 0; i < 4; i++) {
    M1[3 * i    ] = as[i] * fu;
    M1[3 * i + 1] = 0.0;
    M1[3 * i + 2] = as[i] * (uc - u);

    M2[3 * i    ] = 0.0;
    M2[3 * i + 1] = as[i] * fv;
    M2[3 * i + 2] = as[i] * (vc - v);
  }

  if (!is_weighted)
  {
      return;
  }

    Eigen::Matrix2d Sigma2d;
    double du = uc - u;
    double dv = vc - v;
    Sigma2d(0, 0) = du * du * Sigma3d(2, 2) + 2 * fu * du * Sigma3d(0, 2) + fu * fu * Sigma3d(0, 0);
    Sigma2d(1, 1) = dv * dv * Sigma3d(2, 2) + 2 * fv * dv * Sigma3d(1, 2) + fv * fv * Sigma3d(1, 1);
    Sigma2d(0, 1) = du * dv * Sigma3d(2, 2) + fu * dv * Sigma3d(0, 2) + fv * du * Sigma3d(1, 2) +
                    fu * fv * Sigma3d(0, 1);
    Sigma2d(1, 0) = Sigma2d(0, 1);

    Sigma2d += Eigen::Matrix2d::Identity() * sigma2d * de * de;

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(Sigma2d+Eigen::Matrix2d::Identity() * 1e-2, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector2d svals = svd.singularValues();
    Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
    S(0,0) = 1.0/sqrt(fabs(svals(0)));
    S(1,1) = 1.0/sqrt(fabs(svals(1)));
    Eigen::Matrix2d Sigma2dInv_sqrt = svd.matrixU() * S * svd.matrixU().transpose();
    Eigen::Map<Eigen::Matrix<double, 2, 12, Eigen::RowMajor> > M_pair(M1);
    M_pair = Sigma2dInv_sqrt * M_pair;

}

void PnPUsolver2::compute_ccs(const double * betas, const double * ut)
{
  for(int i = 0; i < 4; i++)
    ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

  for(int i = 0; i < 4; i++) {
    const double * v = ut + 12 * (11 - i);
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
	ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

void PnPUsolver2::compute_pcs(void)
{
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = alphas + 4 * i;
    double * pc = pcs + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

double PnPUsolver2::compute_pose(double R[3][3], double t[3], bool is_minimal, const Eigen::Matrix3d& Rest,
        const Eigen::Vector3d& test)
{
  choose_control_points(is_minimal);
  compute_barycentric_coordinates();

  CvMat * M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);

    for(int i = 0; i < number_of_correspondences; i++) {
        double de = 0;
        if (!is_minimal) {
            cv::Point3f pt3d = mvP3Dw[mvChosenIndices[i]];
            Eigen::Vector3d X(pt3d.x, pt3d.y, pt3d.z);
            Eigen::Vector3d Xc = Rest * X + test;
            de = Xc(2);
        }
        fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1], Rest*Sigmas3d[mvChosenIndices[i]]*Rest.transpose(),
               !is_minimal, de, mvSigma2[mvChosenIndices[i]]);
    }

  double mtm[12 * 12], d[12], ut[12 * 12];
  CvMat MtM = cvMat(12, 12, CV_64F, mtm);
  CvMat D   = cvMat(12,  1, CV_64F, d);
  CvMat Ut  = cvMat(12, 12, CV_64F, ut);

  cvMulTransposed(M, &MtM, 1);
  cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
  cvReleaseMat(&M);

  double l_6x10[6 * 10], rho[6];
  CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
  CvMat Rho    = cvMat(6,  1, CV_64F, rho);

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

void PnPUsolver2::copy_R_and_t(const double R_src[3][3], const double t_src[3],
			double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

double PnPUsolver2::dist2(const double * p1, const double * p2)
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPUsolver2::dot(const double * v1, const double * v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

double PnPUsolver2::reprojection_error(const double R[3][3], const double t[3])
{
  double sum2 = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    double * pw = pws + 3 * i;
    double Xc = dot(R[0], pw) + t[0];
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
    double ue = uc + fu * Xc * inv_Zc;
    double ve = vc + fv * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];

    sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
  }

  return sum2 / number_of_correspondences;
}

void PnPUsolver2::estimate_R_and_t(double R[3][3], double t[3])
{
  double pc0[3], pw0[3];

  pc0[0] = pc0[1] = pc0[2] = 0.0;
  pw0[0] = pw0[1] = pw0[2] = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    const double * pc = pcs + 3 * i;
    const double * pw = pws + 3 * i;

    for(int j = 0; j < 3; j++) {
      pc0[j] += pc[j];
      pw0[j] += pw[j];
    }
  }
  for(int j = 0; j < 3; j++) {
    pc0[j] /= number_of_correspondences;
    pw0[j] /= number_of_correspondences;
  }

  double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
  CvMat ABt   = cvMat(3, 3, CV_64F, abt);
  CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
  CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
  CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

  cvSetZero(&ABt);
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pc = pcs + 3 * i;
    double * pw = pws + 3 * i;

    for(int j = 0; j < 3; j++) {
      abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
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

void PnPUsolver2::print_pose(const double R[3][3], const double t[3])
{
  cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
  cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
  cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

void PnPUsolver2::solve_for_sign(void)
{
  if (pcs[2] < 0.0) {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
	ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

double PnPUsolver2::compute_R_and_t(const double * ut, const double * betas,
			     double R[3][3], double t[3])
{
  compute_ccs(betas, ut);
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void PnPUsolver2::find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x4[6 * 4], b4[4];
  CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
  CvMat B4    = cvMat(4, 1, CV_64F, b4);

  for(int i = 0; i < 6; i++) {
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

void PnPUsolver2::find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x3[6 * 3], b3[3];
  CvMat L_6x3  = cvMat(6, 3, CV_64F, l_6x3);
  CvMat B3     = cvMat(3, 1, CV_64F, b3);

  for(int i = 0; i < 6; i++) {
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

void PnPUsolver2::find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x5[6 * 5], b5[5];
  CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
  CvMat B5    = cvMat(5, 1, CV_64F, b5);

  for(int i = 0; i < 6; i++) {
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

void PnPUsolver2::compute_L_6x10(const double * ut, double * l_6x10)
{
  const double * v[4];

  v[0] = ut + 12 * 11;
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 *  9;
  v[3] = ut + 12 *  8;

  double dv[4][6][3];

  for(int i = 0; i < 4; i++) {
    int a = 0, b = 1;
    for(int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
	a++;
	b = a + 1;
      }
    }
  }

  for(int i = 0; i < 6; i++) {
    double * row = l_6x10 + 10 * i;

    row[0] =        dot(dv[0][i], dv[0][i]);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
    row[2] =        dot(dv[1][i], dv[1][i]);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
    row[5] =        dot(dv[2][i], dv[2][i]);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
    row[9] =        dot(dv[3][i], dv[3][i]);
  }
}

void PnPUsolver2::compute_rho(double * rho)
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

void PnPUsolver2::compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
					double betas[4], CvMat * A, CvMat * b)
{
  for(int i = 0; i < 6; i++) {
    const double * rowL = l_6x10 + i * 10;
    double * rowA = A->data.db + i * 4;

    rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
    rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
    rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
    rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

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

void PnPUsolver2::gauss_newton(const CvMat * L_6x10, const CvMat * Rho,
			double betas[4])
{
  const int iterations_number = 5;

  double a[6*4], b[6], x[4];
  CvMat A = cvMat(6, 4, CV_64F, a);
  CvMat B = cvMat(6, 1, CV_64F, b);
  CvMat X = cvMat(4, 1, CV_64F, x);

  for(int k = 0; k < iterations_number; k++) {
    compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db,
				 betas, &A, &B);
    qr_solve(&A, &B, &X);

    for(int i = 0; i < 4; i++)
      betas[i] += x[i];
  }
}

void PnPUsolver2::qr_solve(CvMat * A, CvMat * b, CvMat * X)
{
  static int max_nr = 0;
  static double * A1, * A2;

  const int nr = A->rows;
  const int nc = A->cols;

  if (max_nr != 0 && max_nr < nr) {
    delete [] A1;
    delete [] A2;
  }
  if (max_nr < nr) {
    max_nr = nr;
    A1 = new double[nr];
    A2 = new double[nr];
  }

  double * pA = A->data.db, * ppAkk = pA;
  for(int k = 0; k < nc; k++) {
    double * ppAik = ppAkk, eta = fabs(*ppAik);
    for(int i = k + 1; i < nr; i++) {
      double elt = fabs(*ppAik);
      if (eta < elt) eta = elt;
      ppAik += nc;
    }

    if (eta == 0) {
      A1[k] = A2[k] = 0.0;
      cerr << "God damnit, A is singular, this shouldn't happen." << endl;
      return;
    } else {
      double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
      for(int i = k; i < nr; i++) {
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
      for(int j = k + 1; j < nc; j++) {
	double * ppAik = ppAkk, sum = 0;
	for(int i = k; i < nr; i++) {
	  sum += *ppAik * ppAik[j - k];
	  ppAik += nc;
	}
	double tau = sum / A1[k];
	ppAik = ppAkk;
	for(int i = k; i < nr; i++) {
	  ppAik[j - k] -= tau * *ppAik;
	  ppAik += nc;
	}
      }
    }
    ppAkk += nc + 1;
  }

  // b <- Qt b
  double * ppAjj = pA, * pb = b->data.db;
  for(int j = 0; j < nc; j++) {
    double * ppAij = ppAjj, tau = 0;
    for(int i = j; i < nr; i++)	{
      tau += *ppAij * pb[i];
      ppAij += nc;
    }
    tau /= A1[j];
    ppAij = ppAjj;
    for(int i = j; i < nr; i++) {
      pb[i] -= tau * *ppAij;
      ppAij += nc;
    }
    ppAjj += nc + 1;
  }

  // X = R-1 b
  double * pX = X->data.db;
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
  for(int i = nc - 2; i >= 0; i--) {
    double * ppAij = pA + i * nc + (i + 1), sum = 0;

    for(int j = i + 1; j < nc; j++) {
      sum += *ppAij * pX[j];
      ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];
  }
}



void PnPUsolver2::relative_error(double & rot_err, double & transl_err,
			  const double Rtrue[3][3], const double ttrue[3],
			  const double Rest[3][3],  const double test[3])
{
  double qtrue[4], qest[4];

  mat_to_quat(Rtrue, qtrue);
  mat_to_quat(Rest, qest);

  double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
			 (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
			 (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
			 (qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
			 (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
			 (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
			 (qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  rot_err = min(rot_err1, rot_err2);

  transl_err =
    sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
	 (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
	 (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
    sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

void PnPUsolver2::mat_to_quat(const double R[3][3], double q[4])
{
  double tr = R[0][0] + R[1][1] + R[2][2];
  double n4;

  if (tr > 0.0f) {
    q[0] = R[1][2] - R[2][1];
    q[1] = R[2][0] - R[0][2];
    q[2] = R[0][1] - R[1][0];
    q[3] = tr + 1.0f;
    n4 = q[3];
  } else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) {
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

} //namespace ORB_SLAM
