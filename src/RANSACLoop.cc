/** This is a parent class for a RANSAC loop
* Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
* Redistribution and use is allowed according to the terms of the GPL v3 license.
**/

#include "RANSACLoop.h"

void RANSACLoop::Iterate(cv::Mat img)
{
    int it = 0;
    while (it < Nit)
    {
        int last_inlier_cnt = inlier_cnt;
        if (SolveOnce()) {
            if (last_inlier_cnt != inlier_cnt) {
                RecomputeNIt();
            }
            it++;
        }
    }
    if (!img.empty())
    {
        DrawInliers(img);
    }
}