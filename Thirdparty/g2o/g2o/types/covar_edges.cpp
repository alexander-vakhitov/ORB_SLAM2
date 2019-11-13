//
// Created by alexander on 25.10.19.
//

#include <Eigen/Dense>

#include "types_six_dof_expmap.h"

#include "../core/factory.h"
#include "../stuff/macros.h"

namespace g2o {

    using namespace std;

    using namespace Eigen;

    typedef vector<Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d> > mat2dvector;

    Matrix3d cpmat(const Vector3d& s)
    {
        Matrix3d c;
        c.setZero();
        c << 0, -s(2), s(1), s(2), 0, -s(0), -s(1), s(0), 0;
        return c;
    }



    void jacf(const Vector3d& Xc, double fx, double fy, Matrix<double, 2, 3>* jac)
    {
        jac->setZero();
        (*jac)(0, 0) = fx / Xc(2);
        (*jac)(1, 1) = fy / Xc(2);
        (*jac)(0, 2) = -fx * Xc(0) / Xc(2) / Xc(2);
        (*jac)(1, 2) = -fy * Xc(1) / Xc(2) / Xc(2);
    }

    void jac_stereo_f(const Vector3d& Xc, double fx, double fy, double b, Matrix3d* jac)
    {
        jac->setZero();
        (*jac)(0, 0) = fx / Xc(2);
        (*jac)(1, 1) = fy / Xc(2);
        (*jac)(0, 2) = -fx * Xc(0) / Xc(2) / Xc(2);
        (*jac)(1, 2) = -fy * Xc(1) / Xc(2) / Xc(2);
        (*jac)(2, 0) = fx / Xc(2);
        (*jac)(2, 2) = -fx * (Xc(0)-b) / Xc(2) / Xc(2);
    }

    void xc_d_p(const Vector3d& Xc, Matrix<double, 3, 6>* xc_d_p)
    {
        xc_d_p->block<3,3>(0, 0) = -cpmat(Xc);
        xc_d_p->block<3,3>(0, 3) = Matrix3d::Identity();
    }

    void jac_d_p(const Vector3d& Xc, double fx, double fy, vector< Matrix<double, 2, 3> >* jac_dp)
    {
        jac_dp->clear();
        Matrix<double, 2, 3> jac_d_xc1, jac_d_xc2, jac_d_xc3;
        jac_d_xc1.setZero();
        jac_d_xc1(0, 2) = -fx / Xc(2) / Xc(2);

        jac_d_xc2.setZero();
        jac_d_xc2(1, 2) = -fy / Xc(2) / Xc(2);

        jac_d_xc3.setZero();
        jac_d_xc3(0, 0) = -fx / Xc(2) / Xc(2);
        jac_d_xc3(1, 1) = -fy / Xc(2) / Xc(2);
        jac_d_xc3(0, 2) = 2.0 * fx * Xc(0) / Xc(2) / Xc(2) / Xc(2);
        jac_d_xc3(1, 2) = 2.0 * fy * Xc(1) / Xc(2) / Xc(2) / Xc(2);

        Matrix<double, 3, 6> xc_dp;
        xc_d_p(Xc, &xc_dp);
        for (int p = 0; p < 6; p++)
        {
            jac_dp->push_back(jac_d_xc1 * xc_dp(0, p) + jac_d_xc2 * xc_dp(1, p) + jac_d_xc3 * xc_dp(2, p));
        }
    }

    void jac_stereo_d_p(const Vector3d& Xc, double fx, double fy, double b, vector< Matrix<double, 3, 3> >* jac_dp)
    {
        jac_dp->clear();
        Matrix<double, 3, 3> jac_d_xc1, jac_d_xc2, jac_d_xc3;
        jac_d_xc1.setZero();
        jac_d_xc1(0, 2) = -fx / Xc(2) / Xc(2);
        jac_d_xc1(2, 2) = -fx / Xc(2) / Xc(2);

        jac_d_xc2.setZero();
        jac_d_xc2(1, 2) = -fy / Xc(2) / Xc(2);

        jac_d_xc3.setZero();
        jac_d_xc3(0, 0) = -fx / Xc(2) / Xc(2);
        jac_d_xc3(1, 1) = -fy / Xc(2) / Xc(2);
        jac_d_xc3(0, 2) = 2.0 * fx * Xc(0) / Xc(2) / Xc(2) / Xc(2);
        jac_d_xc3(1, 2) = 2.0 * fy * Xc(1) / Xc(2) / Xc(2) / Xc(2);
        jac_d_xc3(2, 0) = -fx / Xc(2) / Xc(2);
        jac_d_xc3(2, 2) = 2.0 * fx * (Xc(0)-b) / Xc(2) / Xc(2) / Xc(2);

        Matrix<double, 3, 6> xc_dp;
        xc_d_p(Xc, &xc_dp);
        for (int p = 0; p < 6; p++)
        {
            jac_dp->push_back(jac_d_xc1 * xc_dp(0, p) + jac_d_xc2 * xc_dp(1, p) + jac_d_xc3 * xc_dp(2, p));
        }
    }


    void rot_d_p(const Matrix3d& R, vector<Matrix3d>* dR)
    {
        Vector3d vec_i(1,0,0);
        Vector3d vec_j(0,1,0);
        Vector3d vec_k(0,0,1);
        Matrix3d dRi = -cpmat(vec_i);
        Matrix3d dRj = -cpmat(vec_j);
        Matrix3d dRk = -cpmat(vec_k);
        for (int q = 0; q < 6; q++)
        {
            dR->push_back(Matrix3d::Zero());
        }
        for (int q = 0; q < 3; q++)
        {
            (*dR)[q].col(0) = dRi.col(q);
            (*dR)[q].col(1) = dRj.col(q);
            (*dR)[q].col(2) = dRk.col(q);
        }
        for (int q = 0; q < 3; q++)
        {
            (*dR)[q] = (*dR)[q] * R;
        }
    }

//    function dM = cov_d_p(R, t, X, sigma_p_2)
//    Xc = R*X + t;
//    j_d_p = jac_d_p(Xc);
//    rot_d_p = R_d_p(R);
//    jac = jacf(Xc);
//    dM = zeros(2, 2, 6);
//    for p = 1:6
//    dMp = j_d_p(:, :, p) * R * sigma_p_2 * R' * jac';
//    dMp = dMp + jac * rot_d_p(:, :, p) * sigma_p_2 * R' * jac';
//    dMp = dMp + jac * R * sigma_p_2 * rot_d_p(:, :, p)' * jac';
//    dMp = dMp + jac * R * sigma_p_2 * R' * j_d_p(:, :, p)';
//    dM(:, :, p) = dMp;
//    end
//            end

    void cov_d_p(const Matrix3d& R, const Vector3d& t, const Vector3d& X, const Matrix3d& sigma_p_2,
                 const double sigma_d_2, double fx, double fy, mat2dvector* dM, Matrix2d* Sigma)
    {
        Vector3d Xc = R * X + t;
        vector< Matrix<double, 2, 3> > jac_dp;
        jac_d_p(Xc, fx, fy, &jac_dp);
        vector<Matrix3d> dR;
        rot_d_p(R, &dR);
        Matrix<double, 2, 3> jac;
        jacf(Xc, fx,fy, &jac);
        dM->clear();
        for (int i = 0; i < 6; i++)
        {
            Matrix2d dMp = jac_dp[i] * R * sigma_p_2 * R.transpose() * jac.transpose();
            dMp = dMp + jac * dR[i] * sigma_p_2 * R.transpose() * jac.transpose();
            dMp = dMp + jac * R * sigma_p_2 * dR[i].transpose() * jac.transpose();
            dMp = dMp + jac * R * sigma_p_2 * R.transpose() * jac_dp[i].transpose();
//            Matrix2d dMp = jac_dp[i] * sigma_p_2 * jac.transpose();
//            dMp = dMp + jac * sigma_p_2 * jac_dp[i].transpose();
            dM->push_back(dMp);
        }
        //Xc = R * X + t;
        //    jac = jacf(Xc);
        //
        //    Sigma = sigma_d_2 * eye(2) + jac * R * sigma_p_2 * R' * jac';
        *Sigma = sigma_d_2 * Matrix2d::Identity() + jac * R * sigma_p_2 * R.transpose() * jac.transpose();
    }

    void cov_stereo_d_p(const Matrix3d& R, const Vector3d& t, const Vector3d& X, const Matrix3d& sigma_p_2,
                 const double sigma_d_2, double fx, double fy, double b, vector<Matrix3d>* dM, Matrix3d* Sigma)
    {
        Vector3d Xc = R * X + t;
        vector< Matrix<double, 3, 3> > jac_dp;
        jac_stereo_d_p(Xc, fx, fy, b, &jac_dp);
        vector<Matrix3d> dR;
        rot_d_p(R, &dR);
        Matrix3d jac;
        jac_stereo_f(Xc, fx, fy, b, &jac);
        dM->clear();
        for (int i = 0; i < 6; i++)
        {
            Matrix3d dMp = jac_dp[i] * R * sigma_p_2 * R.transpose() * jac.transpose();
            dMp = dMp + jac * dR[i] * sigma_p_2 * R.transpose() * jac.transpose();
            dMp = dMp + jac * R * sigma_p_2 * dR[i].transpose() * jac.transpose();
            dMp = dMp + jac * R * sigma_p_2 * R.transpose() * jac_dp[i].transpose();
            dM->push_back(dMp);
        }
        //Xc = R * X + t;
        //    jac = jacf(Xc);
        //
        //    Sigma = sigma_d_2 * eye(2) + jac * R * sigma_p_2 * R' * jac';
        *Sigma = sigma_d_2 * Matrix3d::Identity() + jac * R * sigma_p_2 * R.transpose() * jac.transpose();
    }

    template <typename M> void svd_symm_jac(const M& A, const M& dA, const M& U, const M& S, M* dU_p, M* dS_p)
    {
        int q = A.cols();
        M F;
        F.setZero();
        bool is_eq_eig = false;
        for (int i = 0; i < q; i++)
        {
            for (int j = 0; j < q; j++)
            {
                if (i == j)
                {
                    continue;
                }
                double svi = S(i, i);
                double svj = S(j, j);
                if (svi == svj)
                {
                    is_eq_eig = true;
                    continue;
                }
                F(i, j) = 1.0 / ( svj * svj - svi * svi );
            }
        }
        M id;
        id.setIdentity();

        if (is_eq_eig)
        {
            dU_p->setZero();
        } else {
            M dU_add = (id - U * U.transpose()) * dA * U * S.inverse();
//            std::cout << "du add norm " << dU_add.norm() << std::endl;
            M dU_main = U.transpose() * dA * U * S + S * U.transpose() * dA.transpose() * U;
//            std::cout << "du main 1 norm " << dU_main.norm() << std::endl;
            dU_main = (F.array() * dU_main.array()).matrix();
//            std::cout << "du main 2 norm " << dU_main.norm() << std::endl;
            dU_main = U * dU_main;
//            std::cout << "du main 3 norm " << dU_main.norm() << std::endl;
            *dU_p = dU_main + dU_add;
        }
//        std::cout << "du p norm " << dU_p->norm() << std::endl;
        *dS_p = (id.array() * (U.transpose() * dA * U).array()).matrix();
    }

    void invsqrt_cov_d_p(const Matrix3d& R, const Vector3d& t, const Vector3d& X,
            const Matrix3d& sigma_p_2, double sigma_d_2, double fx, double fy,
            mat2dvector* dM_p, Matrix2d* M)
    {
        mat2dvector dM;
        Matrix2d Sigma;
        cov_d_p(R, t, X, sigma_p_2, sigma_d_2, fx, fy, &dM, &Sigma);

//        *dM_p = dM;
//        return;

        Matrix2d U, S;
        eigen_safe(Sigma, &U, &S);
        Vector2d svals = S.diagonal();
//        S.diagonal() << svals[0], svals[1];
        DiagonalMatrix<double, 2> Si;
        Si.diagonal() << 1.0 / sqrt(fabs(svals(0))), 1.0/sqrt(fabs(svals(1)));
        *M = U * Si * U.transpose();
        DiagonalMatrix<double, 2> Sii;
        Sii.diagonal() << 1.0 / sqrt(fabs(svals(0))) / fabs(svals(0)), 1.0 / sqrt(fabs(svals(1))) / fabs(svals(1));
        Sii =  -0.5 * Sii;
        for (int q = 0; q < 6; q++)
        {
            Matrix2d dU, dS;
            Matrix2d Sd = S;
            svd_symm_jac(Sigma, dM[q], U, Sd, &dU, &dS);
            Matrix2d dMi = dU * Si * U.transpose() + U * Sii * dS * U.transpose() + U * Si * dU.transpose();
//            std::cout << "svd_symm_jac " << q << " norms " << dMi.norm() <<  " " << dU.norm() << " " << dS.norm() << " "<< std::endl;
            dM[q] = dMi;
        }
        *dM_p = dM;
    }

    template <typename M> void eigen_safe(const M& A, M* U, M* S)
    {
        SelfAdjointEigenSolver<M> es(A);
        *U = es.eigenvectors().real();
        int q = U->cols();
        for (int i = 0; i < q; i++)
        {
            if ((*U)(0, i) < 0)
            {
                U->col(i) = -1 * U->col(i);
            }
        }
        *S = es.eigenvalues().asDiagonal();
    }

    template void eigen_safe<Matrix2d>(const Matrix2d&, Matrix2d*, Matrix2d*);

    template void eigen_safe<Matrix3d>(const Matrix3d&, Matrix3d*, Matrix3d*);

    void invsqrt_cov_stereo_d_p(const Matrix3d& R, const Vector3d& t, const Vector3d& X,
                         const Matrix3d& sigma_p_2, double sigma_d_2, double fx, double fy,
                         double b, vector<Matrix3d>* dM_p, Matrix3d* M)
    {
        vector<Matrix3d> dM;
        Matrix3d Sigma;
        cov_stereo_d_p(R, t, X, sigma_p_2, sigma_d_2, fx, fy, b, &dM, &Sigma);

//        *dM_p = dM;
//        return;

        Matrix3d U, S;
        eigen_safe(Sigma, &U, &S);
        Vector3d svals = S.diagonal();
        DiagonalMatrix<double, 3> Si;
        Si.diagonal() << 1.0 / sqrt(fabs(svals(0))), 1.0/sqrt(fabs(svals(1))), 1.0/sqrt(fabs(svals(2)));
        *M = U * Si * U.transpose();
        DiagonalMatrix<double, 3> Sii;
        Sii.diagonal() << 1.0 / sqrt(fabs(svals(0))) / fabs(svals(0)),
                          1.0 / sqrt(fabs(svals(1))) / fabs(svals(1)),
                          1.0 / sqrt(fabs(svals(2))) / fabs(svals(2));
        Sii =  -0.5 * Sii;
        for (int q = 0; q < 6; q++)
        {
            Matrix3d dU, dS;
            Matrix3d Sd = S;
            svd_symm_jac(Sigma, dM[q], U, Sd, &dU, &dS);
//            std::cout << "svd_symm_jac done " << dM.size() << std::endl;
            Matrix3d dMi = dU * Si * U.transpose() + U * Sii * dS * U.transpose() + U * Si * dU.transpose();
            dM[q] = dMi;
        }
        *dM_p = dM;
    }


    void EdgeSE3CovProjectXYZOnlyPose::linearizeOplus()
    {
//        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
//        Matrix3d R = v1->estimate().rotation().toRotationMatrix();
//        Vector3d t = v1->estimate().translation();
//        mat2dvector dM;
//        Matrix2d M;
//        invsqrt_cov_d_p(R, t, Xw, Sigma_p_2, sigma_d_2, fx, fy, &dM, &M);
////        std::cout << "invsqrt_cov_d_p done " << std::endl;
//        Vector3d Xc = v1->estimate().map(Xw);
//        Vector2d obs(_measurement);
//
//        double x = Xc[0];
//        double y = Xc[1];
//        double invz = 1.0/Xc[2];
//        double invz_2 = invz*invz;
//
//        Vector2d u = obs - cam_project(Xc);
//
//        _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
//        _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
//        _jacobianOplusXi(0,2) = y*invz *fx;
//        _jacobianOplusXi(0,3) = -invz *fx;
//        _jacobianOplusXi(0,4) = 0;
//        _jacobianOplusXi(0,5) = x*invz_2 *fx;
//
//        _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
//        _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
//        _jacobianOplusXi(1,2) = -x*invz *fy;
//        _jacobianOplusXi(1,3) = 0;
//        _jacobianOplusXi(1,4) = -invz *fy;
//        _jacobianOplusXi(1,5) = y*invz_2 *fy;
//
//        _jacobianOplusXi = M * _jacobianOplusXi;
//
////        std::cout << "M mono unit mat diff " << (M - Matrix2d::Identity()).norm() << std::endl;
//        for (int q = 0; q < 6; q++)
//        {
//            _jacobianOplusXi.col(q) = _jacobianOplusXi.col(q) + dM[q] * u;
//        }

        Matrix<double, 2, 6> jac;
        compute_jacobian(&jac);
        _jacobianOplusXi = jac;
    }

    void EdgeStereoSE3CovProjectXYZOnlyPose::linearizeOplus()
    {
//        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
//        Matrix3d R = v1->estimate().rotation().toRotationMatrix();
//        Vector3d t = v1->estimate().translation();
//        vector<Matrix3d> dM;
//        Matrix3d M;
//        invsqrt_cov_stereo_d_p(R, t, Xw, Sigma_p_2, sigma_d_2, fx, fy, b, &dM, &M);
////        std::cout << "invsqrt_cov_d_p done " << std::endl;
//        Vector3d Xc = v1->estimate().map(Xw);
//        Vector3d obs(_measurement);
//
//        double x = Xc[0];
//        double y = Xc[1];
//        double invz = 1.0/Xc[2];
//        double invz_2 = invz*invz;
//
//        Vector3d u = obs - cam_project(Xc);
//
//        _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
//        _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
//        _jacobianOplusXi(0,2) = y*invz *fx;
//        _jacobianOplusXi(0,3) = -invz *fx;
//        _jacobianOplusXi(0,4) = 0;
//        _jacobianOplusXi(0,5) = x*invz_2 *fx;
//
//        _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
//        _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
//        _jacobianOplusXi(1,2) = -x*invz *fy;
//        _jacobianOplusXi(1,3) = 0;
//        _jacobianOplusXi(1,4) = -invz *fy;
//        _jacobianOplusXi(1,5) = y*invz_2 *fy;
//
//        _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-b*fx*y*invz_2;
//        _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)+b*fx*x*invz_2;
//        _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2);
//        _jacobianOplusXi(2,3) = _jacobianOplusXi(0,3);
//        _jacobianOplusXi(2,4) = 0;
//        _jacobianOplusXi(2,5) = _jacobianOplusXi(0,5)-b*fx*invz_2;
//
//        _jacobianOplusXi = M * _jacobianOplusXi;
//
////        std::cout << "M stereo unit mat diff " << (M - Matrix3d::Identity()).norm() << std::endl;
//        for (int q = 0; q < 6; q++)
//        {
//            _jacobianOplusXi.col(q) = _jacobianOplusXi.col(q) + dM[q] * u;
//        }
        Matrix<double, 3, 6> jac;
        compute_jacobian(&jac);
        _jacobianOplusXi = jac;
    }

    void EdgeSE3CovProjectXYZOnlyPose::compute_jacobian(Matrix<double, 2, 6>* jac)
    {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        Matrix3d R = v1->estimate().rotation().toRotationMatrix();
        Vector3d t = v1->estimate().translation();
        Vector3d Xc = v1->estimate().map(Xw);
        double x = Xc[0];
        double y = Xc[1];
        double invz = 1.0/Xc[2];
        double invz_2 = invz*invz;

        (*jac)(0,0) =  x*y*invz_2 *fx;
        (*jac)(0,1) = -(1+(x*x*invz_2)) *fx;
        (*jac)(0,2) = y*invz *fx;
        (*jac)(0,3) = -invz *fx;
        (*jac)(0,4) = 0;
        (*jac)(0,5) = x*invz_2 *fx;

        (*jac)(1,0) = (1+y*y*invz_2) *fy;
        (*jac)(1,1) = -x*y*invz_2 *fy;
        (*jac)(1,2) = -x*invz *fy;
        (*jac)(1,3) = 0;
        (*jac)(1,4) = -invz *fy;
        (*jac)(1,5) = y*invz_2 *fy;

        if (use_sigma_est)
        {
            *jac = Sigma_est_sqin * (*jac);
            return;
        }

        mat2dvector dM;
        Matrix2d M;
        invsqrt_cov_d_p(R, t, Xw, Sigma_p_2, sigma_d_2, fx, fy, &dM, &M);
//        std::cout << "invsqrt_cov_d_p done " << std::endl;

        Vector2d obs(_measurement);

        Vector2d u = obs - cam_project(Xc);



        (*jac) = M * (*jac);

//        std::cout << "M mono unit mat diff " << (M - Matrix2d::Identity()).norm() << std::endl;
        for (int q = 0; q < 6; q++)
        {
            (*jac).col(q) = (*jac).col(q) + dM[q] * u; //dM[q].col(0);//
        }
    }

    void EdgeStereoSE3CovProjectXYZOnlyPose::compute_jacobian(Matrix<double, 3, 6>* jac)
    {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        Matrix3d R = v1->estimate().rotation().toRotationMatrix();
        Vector3d t = v1->estimate().translation();

        Vector3d Xc = v1->estimate().map(Xw);
        Vector3d obs(_measurement);

        double x = Xc[0];
        double y = Xc[1];
        double invz = 1.0/Xc[2];
        double invz_2 = invz*invz;

        Vector3d u = obs - cam_project(Xc);

        (*jac)(0,0) =  x*y*invz_2 *fx;
        (*jac)(0,1) = -(1+(x*x*invz_2)) *fx;
        (*jac)(0,2) = y*invz *fx;
        (*jac)(0,3) = -invz *fx;
        (*jac)(0,4) = 0;
        (*jac)(0,5) = x*invz_2 *fx;

        (*jac)(1,0) = (1+y*y*invz_2) *fy;
        (*jac)(1,1) = -x*y*invz_2 *fy;
        (*jac)(1,2) = -x*invz *fy;
        (*jac)(1,3) = 0;
        (*jac)(1,4) = -invz *fy;
        (*jac)(1,5) = y*invz_2 *fy;

        (*jac)(2,0) = (*jac)(0,0)-b*fx*y*invz_2;
        (*jac)(2,1) = (*jac)(0,1)+b*fx*x*invz_2;
        (*jac)(2,2) = (*jac)(0,2);
        (*jac)(2,3) = (*jac)(0,3);
        (*jac)(2,4) = 0;
        (*jac)(2,5) = (*jac)(0,5)-b*fx*invz_2;

        if (use_sigma_est)
        {
            *jac = Sigma_est_sqin * (*jac);
            return;
        }

        vector<Matrix3d> dM;
        Matrix3d M;
        invsqrt_cov_stereo_d_p(R, t, Xw, Sigma_p_2, sigma_d_2, fx, fy, b, &dM, &M);
//        std::cout << "invsqrt_cov_d_p done " << std::endl;


        (*jac) = M * (*jac);

//        std::cout << "M stereo unit mat diff " << (M - Matrix3d::Identity()).norm() << std::endl;
        for (int q = 0; q < 6; q++)
        {
            (*jac).col(q) = (*jac).col(q) + dM[q] * u;
        }
    }
}