//
// Created by alexander on 05.11.19.
//

#ifndef SEGO_RUNNER_G2O_POSE_H_H
#define SEGO_RUNNER_G2O_POSE_H_H

#include <Eigen/Dense>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>
#include <opencv/cxeigen.hpp>

typedef g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap > PoseEdge;

void GetJProjOpt(double Xc, double Yc, double Zc, Eigen::Matrix<double, 2, 3>* J_proj_p);
void GetJProjOptStereo(double Xc, double Yc, double Zc, double b, Eigen::Matrix3d* J_proj_p);


class G2O_Pose
{
public:

    G2O_Pose(int refinementType, double delta, const Eigen::Matrix3d& K, const Eigen::Matrix4d &T_init,
            bool is_debug=false, std::string debug_path=""); //0 - standard, 1 - full cov, 2-rotated cov, 3 - truncated cov

    void AddCorrespondence(g2o::SparseOptimizer& optimizer, int id, double X, double Y, double Z, double u, double v, const double s2d,
                                                const double s3d, const Eigen::Matrix3d& S3D, bool is_outlier, double s3dmin=-1, double s3dmax=-1);

    int Refine(Eigen::Matrix4d* T_fin, const std::vector<cv::Point2f>& p2D,
               const std::vector<float>& sigmas2D, const std::vector<cv::Point3f>& p3D, const std::vector<float>& sigmas3D,
               const std::vector<Eigen::Matrix3d>& sigmas3DFull, const std::vector<bool>& vbInliers, double s3dmin=-1, double s3dmax=-1);

private:
    int refinementType;
    double delta;
    Eigen::Matrix3d K;
    Eigen::Matrix4d T_init;
    int nInitialCorrespondences;

    g2o::VertexSE3Expmap * vSE3;

    std::vector<PoseEdge *> edges;
    std::vector<bool> outlier_status;


    bool is_debug;
    std::string debug_path;
    std::ofstream debug_out;
};

#endif //SEGO_RUNNER_G2O_POSE_H_H
