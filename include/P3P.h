//
// Created by alexander on 30.03.17.
//

#ifndef CARPET_POSE_GEOMETRY_UTILS_H
#define CARPET_POSE_GEOMETRY_UTILS_H

#include <opencv2/core/mat.hpp>

void p3p(cv::Mat& det_sample, cv::Mat& templ_sample, std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& ts);

#endif //CARPET_POSE_GEOMETRY_UTILS_H
