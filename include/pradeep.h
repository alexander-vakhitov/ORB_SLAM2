//
// Created by alexander on 14.03.18.
//

#ifndef SM_PRADEEP_H
#define SM_PRADEEP_H
#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

bool pradeep(const cv::Mat& projs, const cv::Mat& lprojs,
                 std::vector<Eigen::Matrix3d>* Rs, std::vector<Eigen::Vector3d>* ts);

void build_C_mat(const Eigen::MatrixXd& A, Eigen::MatrixXd& C);

void build_polys(const Eigen::MatrixXd& B, Eigen::MatrixXd& P);

#endif //SM_PRADEEP_H
