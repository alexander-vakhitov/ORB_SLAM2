//
// Created by alexander on 20.03.19.
//

#ifndef SEGO_RUNNER_PNP3D_NEWTON_H
#define SEGO_RUNNER_PNP3D_NEWTON_H

#include <Eigen/Dense>

void gradient(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* g);
void hessian(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* H);

#endif //SEGO_RUNNER_PNP3D_NEWTON_H
