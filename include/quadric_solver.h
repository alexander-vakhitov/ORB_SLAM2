//
// Created by alexander on 18.01.18.
//

#ifndef SM_QUADRIC_SOLVER_H
#define SM_QUADRIC_SOLVER_H

#include <vector>
#include <Eigen/Dense>

bool solve_3_quadric(const Eigen::Matrix<double, 3, 10>& M, bool is_det_check, std::vector<double>* bs, std::vector<double>* cs, std::vector<double>* ds);

#endif //SM_QUADRIC_SOLVER_H
