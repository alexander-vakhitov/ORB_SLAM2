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

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include "pnp3d.h"

using namespace std;
//#include "MapPoint.h"
//#include "Frame.h"

namespace ORB_SLAM2 {

    class PnPsolver {
    public:
//  PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches);

        PnPsolver(const std::vector<cv::Point2f> &p2D,
                  const std::vector<float> &sigma2, const std::vector<cv::Point3f> &p3D,
                  const std::vector<size_t> &keyPointIndices, const std::vector<size_t> &allIndices,
                  double fu, double fv, double uc, double vc, int nMapPoints,
                  const std::vector<float>& sigmas_3d = std::vector<float>(),
                  const std::vector<Eigen::Matrix3d>& sigmas_3d_full = std::vector<Eigen::Matrix3d>(),
                  int mode = 0);

        ~PnPsolver();

        void SetRansacParameters(double probability = 0.99, int minInliers = 8, int maxIterations = 300, int minSet = 4,
                                 float epsilon = 0.4,
                                 float th2 = 5.991);

        cv::Mat find(vector<bool> &vbInliers, int &nInliers);

        cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

        cv::Mat FinalRefinement(const cv::Mat& T_init, const std::vector<bool>& inliersInit);

        void SwitchMode(int new_mode);

        void SetLines(const std::vector<Eigen::Vector3d>& linesStartPts, const std::vector<Eigen::Vector3d>& linesEndPts,
                      const std::vector<Eigen::Vector3d>& X0s, const std::vector<Eigen::Vector3d>& lineDirs,
                      const vec2d& line_endpts_2d_start,
                      const vec2d& line_endpts_2d_end,
                      const mat6dvector& linesSigmas3D,
                      const std::vector<Eigen::Matrix3d>& linesSigmasProj,
                      const std::vector<Eigen::Matrix3d>& linesSigmasProjNorm,
                      const std::vector<float>& linesDetSigmas2);


        bool GetLineInliers(vector<bool> &vbLineInliers, int &nLineInliers);



    private:

        void CheckInliers();

        void CheckInliersLines();

        bool Refine();

        void AddLineCorrespondences();

        // Functions from the original EPnP code
        void set_maximum_number_of_correspondences(const int n);

        void set_maximum_number_of_line_correspondences(int n);

        void reset_correspondences(void);

        void reset_line_correspondences(void);

        void add_correspondence(const double X, const double Y, const double Z,
                                const double u, const double v, const double s2d, const double s3d,
                                const Eigen::Matrix3d& S3D);

        void add_line_correspondence(const Eigen::Matrix3d& S2d, const Eigen::Matrix3d& S2dN,
                                                const Eigen::Matrix<double,6,6>& S6D,
                                                const Eigen::Vector3d& X_start, const Eigen::Vector3d& X_end,
                                                const Eigen::Vector3d& lineEq, const Eigen::Vector3d& lineEqN,
                                                const Eigen::Vector2d& xs, const Eigen::Vector2d& xe, float detSigma2,
                                     const Eigen::Matrix3d& SigmaLD);

        void fill_M_uncertain_line_2(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, CvMat *M,
                                                const int row, const double *as, const double *ae,
                                                const Eigen::Vector3d& line_eq, const Eigen::Matrix3d& s2d,
                                                const Eigen::Matrix<double,6,6>&S6D, const Eigen::Matrix3d& K,
                                                const Eigen::Vector3d& Xs, const Eigen::Vector3d& Xe,
                                                const Eigen::Vector2d& midPt);

        double compute_pose(double R[3][3], double T[3]);

        double compute_pose_uncertain(const cv::Mat& T_est_cv, double R[3][3], double t[3]);

        double compute_pose_uncertain_accurate(const cv::Mat& T_est_cv, double R[3][3], double t[3]);

        void compute_pose_dlsu(const cv::Mat& T_est_cv, double R[3][3], double t[3], bool is_two_stage=false);

        void compute_pose_dlsu_accurate(const cv::Mat& T_est_cv, double R[3][3], double t[3]);

        void relative_error(double &rot_err, double &transl_err,
                            const double Rtrue[3][3], const double ttrue[3],
                            const double Rest[3][3], const double test[3]);

        void print_pose(const double R[3][3], const double t[3]);

        double reprojection_error(const double R[3][3], const double t[3]);

        void choose_control_points(void);

        void choose_control_points_uncertain(void);

        void choose_control_points_uncertain2();

        void choose_control_points_uncertain_accurate(void);

        void compute_barycentric_coordinates(void);

        void fill_M(CvMat *M, const int row, const double *alphas, const double u, const double v);

        void fill_M_line(CvMat *M, const int row, const double *as, const double *ae,
                const Eigen::Vector3d& line_eq,
                const Eigen::Matrix3d& K);

        void fill_M_line_simple(CvMat *M,
                                           const int row, const double *as, const double *ae,
                                           const Eigen::Vector3d& line_eq,
                                           const Eigen::Matrix3d& K);

        void fill_M_uncertain(const cv::Mat& T_est_cv, CvMat *M, const int row, const double *as, const double u, const double v, const double s2d,
                              const double s3d, double* X);

        void fill_M_uncertain_accurate(const cv::Mat& T_est_cv, CvMat *M,
                                                  const int row, const double *as, const double u, const double v, const double s2d,
                                                  const Eigen::Matrix3d& s3d, double *X);

        void fill_M_uncertain_line(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, CvMat *M,
                                              const int row, const double *as, const double *ae,
                                              const Eigen::Vector3d& line_eq, const Eigen::Matrix3d& s2d,
                                              const Eigen::Matrix<double,6,6>&S6D, const Eigen::Matrix3d& K,
                                              const Eigen::Vector3d& Xs, const Eigen::Vector3d& Xe,
                                                float detSigma2);

        void fill_M_uncertain_line_simple(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, CvMat *M,
                                                     const int row, const double *as, const double *ae,
                                                     const Eigen::Vector3d& line_eq, const Eigen::Matrix3d& s2d,
                                                     const Eigen::Matrix3d& SLD, const Eigen::Matrix3d& K,
                                                     const Eigen::Vector3d& Xs, const Eigen::Vector3d& Xe, float detSigma2);

        void compute_ccs(const double *betas, const double *ut);

        void compute_pcs(void);

        void solve_for_sign(void);

        void find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho, double *betas);

        void find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho, double *betas);

        void find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho, double *betas);

        void qr_solve(CvMat *A, CvMat *b, CvMat *X);

        double dot(const double *v1, const double *v2);

        double dist2(const double *p1, const double *p2);

        void compute_rho(double *rho);

        void compute_L_6x10(const double *ut, double *l_6x10);

        void gauss_newton(const CvMat *L_6x10, const CvMat *Rho, double current_betas[4]);

        void compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                          double cb[4], CvMat *A, CvMat *b);

        double compute_R_and_t(const double *ut, const double *betas,
                               double R[3][3], double t[3]);

        void estimate_R_and_t(double R[3][3], double t[3]);

        void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
                          double R_src[3][3], double t_src[3]);

        void mat_to_quat(const double R[3][3], double q[4]);


        double uc, vc, fu, fv;

        double *pws, *us, *alphas, *pcs;
        double *alphas_start, *alphas_end;
        int maximum_number_of_correspondences, maximum_number_of_line_correspondences;
        int number_of_correspondences, number_of_line_correspondences;
        std::vector<float> sigmas2d_selected;
        std::vector<float> sigmas3d_selected;
        std::vector<Eigen::Matrix3d> Sigmas3D_selected;
        std::vector<float> sigmasDetLineSelected;

        std::vector<Eigen::Matrix3d> Sigmas2dlines_selected, Sigmas2dlinesN_selected;
        mat6dvector Sigmas3dlines_selected;
        std::vector<Eigen::Vector3d> Xs_selected, Xe_selected, lineEqs_selected, lineEqsN_selected;
        vec2d start_pts_2d_selected, end_pts_2d_selected;
        std::vector<Eigen::Matrix3d> Sigmas_LD_selected;

        double cws[4][3], ccs[4][3];
        double cws_determinant;

//  vector<MapPoint*> mvpMapPointMatches;

        // 2D Points
        vector<cv::Point2f> mvP2D;
        vector<float> mvSigma2;

        // 3D Points
        vector<cv::Point3f> mvP3Dw;

        // Index in Frame
        vector<size_t> mvKeyPointIndices;

        // Current Estimation
        double mRi[3][3];
        double mti[3];
        cv::Mat mTcwi;
        vector<bool> mvbInliersi;
        int mnInliersi;

        vector<bool> mvbInliersLines;
        int mnInliersLinesi;

        int mnRefinedInliersLinesi;

        // Current Ransac State
        int mnIterations;
        vector<bool> mvbBestInliers;
        int mnBestInliers;
        cv::Mat mBestTcw;

        // Refined
        cv::Mat mRefinedTcw;
        vector<bool> mvbRefinedInliers;
        int mnRefinedInliers;

        // Number of Correspondences
        int N;

        int N_lines;

        // Indices for random selection [0 .. N-1]
        vector<size_t> mvAllIndices;

        // RANSAC probability
        double mRansacProb;

        // RANSAC min inliers
        int mRansacMinInliers;

        // RANSAC max iterations
        int mRansacMaxIts;

        // RANSAC expected inliers/total ratio
        float mRansacEpsilon;

        // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
        float mRansacTh;

        // RANSAC Minimun Set used at each iteration
        int mRansacMinSet;

        // Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
        vector<float> mvMaxError;

        int nMapPoints;

        std::vector<float> sigmas_3d;
        std::vector<Eigen::Matrix3d> sigmas_3d_full;

        int mode;

        bool bUseLines;

        std::vector<Eigen::Vector3d> mvStartPts, mvEndPts, mvLineX0s, mvLineDirs;
        vec2d mvLineEndpts2dStart,mvLineEndpts2dEnd;
        std::vector<Eigen::Vector3d> mvLineEqs, mvLineEqsNorm;
        mat6dvector mvSigmasLines3D;
        std::vector<Eigen::Matrix3d> mvSigmasLines2D, mvSigmasLines2DNorm;
        std::vector<float> mvMaxLineError;

        float mTh2;

        Eigen::Matrix3d K;
    };

} //namespace ORB_SLAM

#endif //PNPSOLVER_H
