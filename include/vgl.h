//
// Created by alexander on 12.12.17.
//

#ifndef LINEDESCDATACOLLECTOR_VGL_H
#define LINEDESCDATACOLLECTOR_VGL_H
#include <Eigen/Dense>
#include<Eigen/StdVector>

#include <opencv2/core/types.hpp>
#include <opencv2/line_descriptor.hpp>

typedef Eigen::Matrix<double, 3,4> PMatrix;
typedef std::vector<PMatrix, Eigen::aligned_allocator<PMatrix> > projmat_vector;
typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > posevector;
typedef std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d> > vecmat2d;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vec2d;
typedef cv::line_descriptor::KeyLine KeyLine;


namespace vgl {
    bool MultiTriangulateLine(const posevector& Ts,
                                   const std::vector<Eigen::Vector3d>& lines,
                                   Eigen::Vector3d* X0_p, Eigen::Vector3d* line_dir_p);

    bool TriangulateLine(const Eigen::Matrix4d &T1,
                         const Eigen::Matrix4d &T2, const Eigen::Vector3d &line2d_n_1,
                         const Eigen::Vector3d &line2d_n_2, Eigen::Vector3d *X0, Eigen::Vector3d *line_dir);

    bool TriangulateLineAsDepths(const vec2d& epts, const Eigen::Vector3d& l, const Eigen::Matrix3d& Sigma_l,
                                 const vecmat2d& Sigmas_epts, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                                 std::vector<double>* depths, std::vector<Eigen::Matrix3d>* covs, Eigen::Matrix3d* CrossCov);

    void NormalizedLineEquation(double sx, double sy, double ex, double ey, const Eigen::Matrix3d& K, Eigen::Vector3d* lineEq);

    //line is represented as [a,b,x,y]: n = [cos(a)cos(b), cos(a)sin(b), sin(a)] is a normal to the plane Pi orthogonal
    // to a line and passing through the origin, (x,y) is coordinate of intersection point between the line and the plane,
    // coords in the plane are defined as orthogonal with axis X being a projection of the z axis (if zero, then y, then x),
    // Y axis is the cross product between normal and X axis
    void Line3DFromPluecker(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, int* ax_code, double* coords);
    void PlueckerFromLine3D(const double coords[4], int ax_code, Eigen::Vector3d* X0, Eigen::Vector3d* line_dir);

    void RLine3DFromPluecker(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, double* coords);
    void PlueckerFromRLine3D(const double coords[4], Eigen::Vector3d* X0, Eigen::Vector3d* line_dir, Eigen::Matrix4d* J_X0 = NULL, Eigen::Matrix4d* J_line_dir = NULL);

    bool IsLineCrossingRect(const cv::Point2f& lu, const cv::Point2f& rd, const Eigen::Vector3d& line_eq, std::vector<Eigen::Vector3d>* pps, bool segment_priority=false);

    bool IsLineProjectedToCamera(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, const Eigen::Matrix3d& K,
                                 const Eigen::Matrix4d& T, const cv::Size& im_size, Eigen::Vector3d* line_eq_unnormed, std::vector<Eigen::Vector3d>* pps);

    void ReprojectLinePointTo3D(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, const Eigen::Vector2d& pp, const Eigen::Matrix3d& K,
                                double* depth, double* line_param);

    void ProjectLine(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, const Eigen::Matrix4d& T, Eigen::Vector3d* line_eq);

    double LineReprojErrorL1(const Eigen::Vector2d& xs, const Eigen::Vector2d& xe, const Eigen::Matrix<double, 4, 4>& T_other, const Eigen::Vector3d& X0,
                                  const Eigen::Vector3d& line_dir, const Eigen::Matrix3d& K);

    double LineEptReprojError(const Eigen::Vector3d& l, const Eigen::Matrix<double, 4, 4>& T, const Eigen::Vector3d& X1, const Eigen::Vector3d& X2, double f);

//    bool ReprojBehind(const Eigen::Vector3d& xs3, const Eigen::Vector3d& xe3, const Eigen::Matrix<double, 4, 4>& T_other, const Eigen::Vector3d& X0,
//                 const Eigen::Vector3d& line_dir);

//    bool RefineLineStereo(const KeyLine& kl1, const KeyLine& kl2, const cv::Mat& frame1, const cv::Mat& frame2, KeyLine* kl1_ref_p, KeyLine* kl2_ref_p);

    void ReprojectEndpointTo3D(const Eigen::Vector3d& endpoint, const Eigen::Vector3d& X0, const Eigen::Vector3d& d_rot, double* p);


    void LineFrom3DEndpoints(const Eigen::Vector3d& X1, const Eigen::Vector3d& X2, Eigen::Vector3d* X0, Eigen::Vector3d* line_dir);

    Eigen::Vector3d MapPoint(const Eigen::Matrix4d& T_c2w, const Eigen::Vector3d& X);

    Eigen::Vector2d ProjectPoint(const Eigen::Matrix3d& K, Eigen::Vector3d& X);

    void EncodeLineMinimal(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, Eigen::Matrix3d* R_line_p, double* alpha);

    void DecodeLineMinimal(const Eigen::Matrix3d& R, double alpha, Eigen::Vector3d* X0, Eigen::Vector3d* line_dir);

    Eigen::Matrix3d cpmat(const Eigen::Vector3d& t);

    bool TriangulateLineProjective(const PMatrix& P1,
                                   const PMatrix& P2,
                                   const vec2d& segment1,
                                   const vec2d& segment2,
                                   Eigen::Vector3d* X0, Eigen::Vector3d* line_dir);

    void ReprojectEndpointTo3DProjMat(const PMatrix& P, const Eigen::Vector2d& endpoint, const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, Eigen::Vector3d* endpoint_3d);
}

#endif //LINEDESCDATACOLLECTOR_VGL_H
