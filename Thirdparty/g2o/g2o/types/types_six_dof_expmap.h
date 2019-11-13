// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Modified by Raúl Mur Artal (2014)
// Added EdgeSE3ProjectXYZ (project using focal_length in x,y directions)
// Modified by Raúl Mur Artal (2016)
// Added EdgeStereoSE3ProjectXYZ (project using focal_length in x,y directions)
// Added EdgeSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)
// Added EdgeStereoSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)

#ifndef G2O_SIX_DOF_TYPES_EXPMAP
#define G2O_SIX_DOF_TYPES_EXPMAP

#include "../core/base_vertex.h"
#include "../core/base_binary_edge.h"
#include "../core/base_unary_edge.h"
#include "se3_ops.h"
#include "se3quat.h"
#include "types_sba.h"
#include <Eigen/Geometry>

namespace g2o {
namespace types_six_dof_expmap {
void init();
}

using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;


template <typename M> void eigen_safe(const M& A, M* U, M* S);


/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
class  VertexSE3Expmap : public BaseVertex<6, SE3Quat>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3Expmap();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  virtual void setToOriginImpl() {
    _estimate = SE3Quat();
  }

  virtual void oplusImpl(const double* update_)  {
    Eigen::Map<const Vector6d> update(update_);
    setEstimate(SE3Quat::exp(update)*estimate());
  }
};


class  EdgeSE3ProjectXYZ: public  BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZ();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(v2->estimate()));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }
    

  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
};


class  EdgeStereoSE3ProjectXYZ: public  BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZ();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()),bf);
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector3d cam_project(const Vector3d & trans_xyz, const float &bf) const;

  double fx, fy, cx, cy, bf;
};

class  EdgeSE3ProjectXYZOnlyPose: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw;
  double fx, fy, cx, cy;
};


class  EdgeStereoSE3ProjectXYZOnlyPose: public  BaseUnaryEdge<3, Vector3d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector3d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw;
  double fx, fy, cx, cy, bf;
};

class  EdgeStereoSE3CovProjectXYZOnlyPose: public  BaseUnaryEdge<3, Vector3d, VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeStereoSE3CovProjectXYZOnlyPose(){}

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void compute_jacobian(Matrix<double, 3, 6>* jac);

    void computeError()
    {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        Vector3d obs(_measurement);
        Vector3d Xc = v1->estimate().map(Xw);
        _error = obs - cam_project(Xc);

        if (use_sigma_est)
        {
            _error = Sigma_est_sqin * _error;
            return;
        }

        Matrix3d J;
        J.setZero();
        J(0, 0) = fx * 1.0 / Xc(2);
        J(0, 2) = -fx * Xc(0) / Xc(2) / Xc(2);
        J(1, 1) = fy * 1.0 / Xc(2);
        J(1, 2) = -fy * Xc(1) / Xc(2) / Xc(2);
        J(2, 0) = fx * 1.0/Xc(2);
        J(2, 2) = -fx * (Xc(0) - b) / Xc(2) / Xc(2);

//        Matrix3d sigma = Matrix3d::Identity() * sigma_d_2 + J * J.transpose() * sigma_p_2;
        Matrix3d R = v1->estimate().rotation().toRotationMatrix();
        Matrix3d Sigma = Matrix3d::Identity() * sigma_d_2 + J * R * Sigma_p_2 * R.transpose() * J.transpose();
        Matrix3d U,S;
        eigen_safe(Sigma, &U, &S);
        Vector3d svals = S.diagonal();

        DiagonalMatrix<double, 3> Si;
        Si.diagonal() << 1.0 / sqrt(fabs(svals(0))), 1.0/sqrt(fabs(svals(1))), 1.0/sqrt(fabs(svals(2)));
        Matrix3d Sigma3dInv_sqrt = U * Si * U.transpose();
        _error = Sigma3dInv_sqrt * _error; //Sigma.col(0);//
    }

    bool isDepthPositive() {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        return (v1->estimate().map(Xw))(2)>0.0;
    }

    virtual void linearizeOplus();

    Vector3d cam_project(const Vector3d & trans_xyz) const;

    Vector3d Xw;
    double fx, fy, cx, cy, b;
    double sigma_d_2, sigma_p_2;
    Matrix3d Sigma_p_2, Sigma_est_sqin;
    bool use_sigma_est = false;
};

class  EdgeSE3CovProjectXYZOnlyPose: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeSE3CovProjectXYZOnlyPose(){}

        bool read(std::istream& is);

        bool write(std::ostream& os) const;

        void compute_jacobian(Matrix<double, 2, 6>* jac);

        void computeError()  {
            const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
            Vector2d obs(_measurement);
            Vector3d Xc = v1->estimate().map(Xw);
            _error = obs-cam_project(Xc);

            if (use_sigma_est)
            {
                _error = Sigma_est_sqin * _error;
                return;
            }

            Matrix<double,2,3> J;
            J.setZero();
            J(0, 0) = fx * 1.0/Xc(2);
            J(0, 2) = -fx * Xc(0) / Xc(2) / Xc(2);
            J(1, 1) = fy * 1.0/Xc(2);
            J(1, 2) = -fy * Xc(1) / Xc(2) / Xc(2);

            Matrix3d R = v1->estimate().rotation().toRotationMatrix();

//            Matrix2d sigma = Matrix2d::Identity() * sigma_d_2 + J * J.transpose() * sigma_p_2;
            Matrix2d Sigma = Matrix2d::Identity() * sigma_d_2 + J * R * Sigma_p_2 * R.transpose() * J.transpose();
//simplified - debug!!!
//            Matrix2d Sigma = Matrix2d::Identity() * sigma_d_2 + J * Sigma_p_2 * J.transpose();
            Matrix2d U,S;
            eigen_safe(Sigma, &U, &S);
            Vector2d svals = S.diagonal();
            DiagonalMatrix<double, 2> Si;
            Si.diagonal() << 1.0 / sqrt(fabs(svals(0))), 1.0/sqrt(fabs(svals(1)));
            Matrix2d Sigma2dInv_sqrt = U * Si * U.transpose();
            _error = Sigma2dInv_sqrt * _error;

//            _error = Sigma.col(0);

        }

        bool isDepthPositive() {
            const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
            return (v1->estimate().map(Xw))(2)>0.0;
        }

        virtual void linearizeOplus();

        Vector2d cam_project(const Vector3d & trans_xyz) const;

        Vector3d Xw;
        double fx, fy, cx, cy;
        double sigma_d_2, sigma_p_2;
        Matrix3d Sigma_p_2;
        Matrix2d Sigma_est_sqin;
        bool use_sigma_est = false;
};


} // end namespace

#endif
