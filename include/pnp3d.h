//
// Created by alexander on 17.08.18.
//

#ifndef TEST_PNP3D_G2O_PNP3D_H
#define TEST_PNP3D_G2O_PNP3D_H

#include <Eigen/Dense>
#include <Eigen/StdVector>

typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> posevector;
typedef std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> mat2dvector;
typedef std::vector<Eigen::Matrix<double,6,6>, Eigen::aligned_allocator<Eigen::Matrix<double,6,6>>> mat6dvector;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vec2d;
typedef std::vector<vec2d, Eigen::aligned_allocator<vec2d>> vecvec2d;

void SolvePnP3D(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                const std::vector<Eigen::Vector3d>& Xs, const std::vector<Eigen::Vector3d>& Xe,
                const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Matrix3d>& S,
                const std::vector<Eigen::Matrix3d>& Ss, const std::vector<Eigen::Matrix3d>& Se,
                const Eigen::Matrix3d& K, posevector* sols_p);//, Eigen::VectorXd& rvec, Eigen::Vector3d& tvec );

void SolvePnPFull(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                const std::vector<Eigen::Vector3d>& Xs, const std::vector<Eigen::Vector3d>& Xe,
                const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Matrix3d>& Sigmas,
                const mat6dvector& SigmasLines, const mat2dvector& Sigmas2D,
                const std::vector<Eigen::Matrix3d>& Sigmas2DLines,
                const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p, int mode = 0);//, Eigen::VectorXd& rvec, Eigen::Vector3d& tvec );

void DLSU(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx_n,
          const std::vector<float>& sigmas3d,
          const std::vector<float>& sigmas2d_norm,
          const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
          bool is_use_3d=true, bool is_use_2d=true);

void RobustDLSU(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
          const std::vector<float>& sigmas3d,
          const std::vector<float>& sigmas2d_norm,
          const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
                bool is_use_3d=true, bool is_use_2d=true);

void DLSU_accurate(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                   const std::vector<Eigen::Matrix3d>& sigmas3d,
                   const std::vector<float>& sigmas2d_norm,
                   const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
                   bool is_use_3d=true, bool is_use_2d=true);

void RobustDLSU_accurate(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                   const std::vector<Eigen::Matrix3d>& sigmas3d,
                   const std::vector<float>& sigmas2d_norm,
                   const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
                   bool is_use_3d=true, bool is_use_2d=true);

void DLSULines(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx_n,
               const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
               const std::vector<Eigen::Vector3d>& l2ds,
               const std::vector<float>& sigmas3d,
               const std::vector<float>& sigmas2d_norm,
               const mat6dvector& sigmas3d_lines,
               const std::vector<float>& sigmasDetLines,
               const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est,
               posevector* sols_p,
               bool is_use_3d=true, bool is_use_2d=true);

void DLSULines_accurate(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx_n,
                        const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
                        const std::vector<Eigen::Vector3d>& l2ds,
                        const std::vector<Eigen::Matrix3d>& sigmas3d,
                        const std::vector<float>& sigmas2d_norm,
                        const mat6dvector& sigmas3d_lines,
                        const std::vector<Eigen::Matrix3d>& sigmas2d_lines,
                        const Eigen::Matrix3d& R_est,
                        const Eigen::Vector3d& t_est, posevector* sols_p);

void DLSLines(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
              const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
              const std::vector<Eigen::Vector3d>& l2ds,
              const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p);

int FindBestSolutionReproj(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                           const Eigen::Matrix3d& K,
                           const posevector& poses, double* min_err_p);

void MLPnP(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
           const std::vector<float>& sigmas2d_norm,
           posevector* sols_p);

#endif //TEST_PNP3D_G2O_PNP3D_H
