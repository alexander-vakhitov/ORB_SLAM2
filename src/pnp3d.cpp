//
// Created by alexander on 17.08.18.
//
#include "pnp3d.h"
#include <Eigen/Dense>
#include <iostream>
#include "pnp3d_newton.h"

#include <opencv2/calib3d.hpp>
#include <opencv/cxeigen.hpp>

using namespace Eigen;

MatrixXcd solver_opt_pnp_hesch2_red(const VectorXd& data);

Matrix3d cpmat(const Vector3d& s)
{
    Matrix3d c;
    c.setZero();
    c << 0, -s(2), s(1), s(2), 0, -s(0), -s(1), s(0), 0;
    return c;
}

void SolvePnP3D(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                const std::vector<Eigen::Vector3d>& Xs, const std::vector<Eigen::Vector3d>& Xe,
                const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Matrix3d>& S,
                const std::vector<Eigen::Matrix3d>& Ss,  const std::vector<Eigen::Matrix3d>& Se,
                const Eigen::Matrix3d& K, posevector* sols_p)//, Eigen::VectorXd& rvec, Eigen::Vector3d& tvec)
{
    int np = XX.size();
    Matrix3d T = Matrix3d::Zero();
    std::vector<Matrix3d> Tp(np, Matrix3d::Zero());
    Matrix<double, 3, 9> A;
    A.setZero();
    for (int i = 0; i < np; i++)
    {
        Vector3d xp(xx[i](0), xx[i](1), 1.0);
        xp = K.inverse() * xp;
        double xsx = xp.transpose() .dot(S[i] * xp);
        Matrix3d Ui = Matrix3d::Identity() - 1.0/xsx*xp*xp.transpose()* S[i];
        Matrix3d USU = Ui.transpose() * S[i] * Ui;
        Tp[i] = USU;
        T = T+USU;
        Vector3d Xp = XX[i];
        Matrix<double, 3, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0,0) = Xp.transpose();
        Ai.block<1,3>(1,3) = Xp.transpose();
        Ai.block<1,3>(2,6) = Xp.transpose();
        A = A + USU*Ai;
    }
    int nl = l.size();
    std::vector<Matrix3d> Tls(nl, Matrix3d::Zero()), Tle(nl, Matrix3d::Zero());
    std::vector<std::vector<Matrix3d>> Tl{Tls, Tle};
    for (int i = 0; i < nl; i++)
    {
        Vector3d li = K.transpose() * l[i];
        li = li / li.norm();
        Matrix3d Li = li * li.transpose();
        std::vector<Vector3d> Xls{Xs[i], Xe[i]};
        std::vector<Matrix3d> Sls{Ss[i], Se[i]};
        for (int j = 0; j < 2; j++)
        {
            Matrix3d LSL = Li*Sls[j]*Li;
            T = T + LSL;
            Tl[j][i] = LSL;
            Vector3d Xl = Xls[j];
            Matrix<double, 3, 9> Ai;
            Ai.setZero();
            Ai.block<1,3>(0,0) = Xl.transpose();
            Ai.block<1,3>(1,3) = Xl.transpose();
            Ai.block<1,3>(2,6) = Xl.transpose();
            A = A + LSL*Ai;
        }
    }
    Matrix<double, 3, 9> A1 = A;

    //Vector3d check = A1*rvec + T*tvec;
    //std::cout << check.transpose() << std::endl;

    Matrix<double, 9, 9> A2;
    A2.setZero();
    for (int i = 0; i < np; i++)
    {
        Vector3d Xp = XX[i];
        Matrix<double, 3, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0,0) = Xp.transpose();
        Ai.block<1,3>(1,3) = Xp.transpose();
        Ai.block<1,3>(2,6) = Xp.transpose();
        Matrix<double, 3, 9> Aim = Ai - T.inverse()*A1;
        A2 = A2 + Aim.transpose() * Tp[i] * Aim;
    }
    for (int i = 0; i < nl; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            Vector3d Xl;
            if (j == 0)
            {
                Xl = Xs[i];
            } else {
                Xl = Xe[i];
            }
            Matrix<double, 3, 9> Ai;
            Ai.setZero();
            Ai.block<1,3>(0,0) = Xl.transpose();
            Ai.block<1,3>(1,3) = Xl.transpose();
            Ai.block<1,3>(2,6) = Xl.transpose();
            Matrix<double, 3, 9> Aim = Ai - T.inverse() * A1;
            A2 = A2 + Aim.transpose() * (Tl[j][i] * Aim);
        }
    }
    Matrix<double, 9, 10> QR;
    QR << 1,0,0,0, 1,0,0,-1, 0, -1,
            0,0,0,-2, 0,2,0,0, 0, 0,
            0,0,2,0, 0,0,2,0, 0, 0,
            0,0,0,2, 0,2,0,0, 0, 0,
            1,0,0,0, -1,0,0,1, 0, -1,
            0,-2,0,0, 0,0,0,0, 2, 0,
            0,0,-2,0, 0,0,2,0, 0, 0,
            0,2,0,0, 0,0,0,0, 2, 0,
            1,0,0,0, -1,0,0,-1, 0, 1;
    Matrix<double, 10, 10> A3 = QR.transpose() * A2 * QR;
    VectorXd data(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            data(10*i+j) = A3(j, i);
        }
    }

    MatrixXcd a = solver_opt_pnp_hesch2_red(data);
    sols_p->clear();
    for (int i = 0; i < a.cols(); i++)
    {
        if (fabs(a.col(i)(0).imag()) == 0)
        {
            Vector3d s = a.col(i).real();
            Matrix3d Rc = (1-s.transpose()*s)*Matrix3d::Identity()+2*cpmat(s)+2*s*s.transpose();
            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            VectorXd rvec(9);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            Vector3d tc = -T.colPivHouseholderQr().solve( A1 * rvec);
            Matrix4d Tc;
            Tc.setIdentity();
            Tc.block<3,3>(0,0) = Rc;
            Tc.block<3,1>(0,3) = tc;
            sols_p->push_back(Tc);
        }
    }
}
// mode = 0 - combined, 1 - 3d, 2 - 2d, 3 - as is
void SolvePnPFull(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                  const std::vector<Eigen::Vector3d>& Xs, const std::vector<Eigen::Vector3d>& Xe,
                  const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Matrix3d>& Sigmas,
                  const mat6dvector& SigmasLines, const mat2dvector& Sigmas2D,
                  const std::vector<Eigen::Matrix3d>& Sigmas2DLines,
                  const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
                  int mode)
{
    int np = XX.size();
    int nl = Xs.size();
    std::vector<double> depths_est;
    std::vector<Eigen::Vector3d> Xsc, Xec;
    std::vector<Eigen::Matrix<double, 2, 3>> Tps;
    std::vector<Eigen::Matrix<double, 3, 3>> Tls;
    Eigen::Matrix3d T;
    T.setZero();
    Eigen::Matrix<double, 3, 9> A;
    A.setZero();
    bool is_est_available = false;
    if (t_est.norm()>0)
    {
        for (int i = 0; i < np; i++)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
        }
        for (int i = 0; i < nl; i++)
        {
            Xsc.push_back(R_est*Xs[i] + t_est);
            Xec.push_back(R_est*Xe[i] + t_est);
        }
        is_est_available = true;
    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix2d Sigma2;
        const Eigen::Matrix3d& S3d = R_est*Sigmas[i]*R_est.transpose();
        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
        Sigma2(0,0) = S3d(2,2)*x(0)*x(0)- 2*S3d(0,2)*x(0)+S3d(0,0);
        Sigma2(1,1) = S3d(2,2)*x(1)*x(1)- 2*S3d(1,2)*x(1)+S3d(1,1);
        Sigma2(0,1) = S3d(2,2)*x(0)*x(1)- S3d(0,2)*x(1)-S3d(1,2)*x(0)+S3d(0,1);
        Sigma2(1,0) = Sigma2(0,1);
        if (is_est_available && mode == 0)
        {
            Sigma2 = Sigma2 + depths_est[i]*depths_est[i]*Sigmas2D[i];
        }
        if (is_est_available  && mode == 2)
        {
            Sigma2 = depths_est[i]*depths_est[i]*Sigmas2D[i];
        }
        if (mode == 3)
        {
            Sigma2.setIdentity();
        }
        Eigen::Matrix2d Sigma2Inv = Sigma2.inverse();
        S2s.push_back(Sigma2Inv);
        Eigen::Matrix<double,2,3> Ti;
        Ti << -1,0, x(0),
              0, -1, x(1);
        Tps.push_back(Ti);
        T = T + Ti.transpose() * Sigma2Inv*Ti;
        const Eigen::Vector3d& Xp = XX[i];
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0, 0) = -Xp.transpose();
        Ai.block<1,3>(0, 6) = x(0)*Xp.transpose();
        Ai.block<1,3>(1, 3) = -Xp.transpose();
        Ai.block<1,3>(1, 6) = x(1)*Xp.transpose();
        Api.push_back(Ai);
        A = A + Ti.transpose()*Sigma2Inv*Ai;
    }
    std::vector<Eigen::Matrix<double, 2, 9>> Ali;
    std::vector<Eigen::Matrix<double, 2, 3>> Tli;
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix2d Sigma2;
        Eigen::Vector3d lvec = K.transpose() * l[i]; // l'X = 0 => l' K X_n = 0 => (K'l)
        lvec /= lvec.segment<2>(0).norm();
        const Eigen::Matrix<double,6,6>& Sl = SigmasLines[i];
        Sigma2(0,0) = lvec.transpose() * R_est * Sl.block<3,3>(0,0) * R_est.transpose() * lvec;
        Sigma2(0,1) = lvec.transpose() * R_est * Sl.block<3,3>(0,3) * R_est.transpose() * lvec;
        Sigma2(1,1) = lvec.transpose() * R_est * Sl.block<3,3>(3,3) * R_est.transpose() * lvec;
        if (is_est_available)
        {
            Sigma2(0,0) = Sigma2(0,0) + Xsc[i].transpose() * Sigmas2DLines[i] * Xsc[i];
            Sigma2(0,1) = Sigma2(0,1) + Xsc[i].transpose() * Sigmas2DLines[i] * Xec[i];
            Sigma2(1,1) = Sigma2(1,1) + Xec[i].transpose() * Sigmas2DLines[i] * Xec[i];
        }
        Sigma2(1,0) = Sigma2(0,1);
        Eigen::Matrix2d Sigma2Inv = Sigma2.inverse();
        S2ls.push_back(Sigma2Inv);
        Eigen::Matrix<double, 2, 3> Ti;
        Ti.row(0) = lvec.transpose();
        Ti.row(1) = lvec.transpose();
        Tli.push_back(Ti);
        const Eigen::Vector3d& Xsi = Xs[i];
        Eigen::Matrix<double, 3, 9> Xsmat;
        Xsmat.setZero();
        Xsmat.block<1,3>(0,0) = Xsi.transpose();
        Xsmat.block<1,3>(1,3) = Xsi.transpose();
        Xsmat.block<1,3>(2,6) = Xsi.transpose();
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.row(0) = lvec.transpose() * Xsmat;
        const Eigen::Vector3d& Xei = Xe[i];
        Xsmat.block<1,3>(0,0) = Xei.transpose();
        Xsmat.block<1,3>(1,3) = Xei.transpose();
        Xsmat.block<1,3>(2,6) = Xei.transpose();
        Ai.row(1) = lvec.transpose() * Xsmat;
        Ali.push_back(Ai);
        Eigen::Matrix<double, 3, 3> Tl = Ti.transpose()*Sigma2Inv*Ti;
        T = T + Tl;
        Tls.push_back(Tl);
        A = A + Ti.transpose() * Sigma2Inv * Ai;
    }
    Eigen::Matrix<double, 9, 9> A2;
    A2.setZero();

    Eigen::Matrix<double, 3,9> T_express = T.colPivHouseholderQr().solve(-A);
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Api[i] + Tps[i]*T_express;
        A2 = A2 + Ai.transpose() * S2s[i] * Ai;
    }
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Ali[i]+Tli[i]*T_express;
        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
    }

    Matrix<double, 9, 10> QR;
    QR << 1,0,0,0, 1,0,0,-1, 0, -1,
            0,0,0,-2, 0,2,0,0, 0, 0,
            0,0,2,0, 0,0,2,0, 0, 0,
            0,0,0,2, 0,2,0,0, 0, 0,
            1,0,0,0, -1,0,0,1, 0, -1,
            0,-2,0,0, 0,0,0,0, 2, 0,
            0,0,-2,0, 0,0,2,0, 0, 0,
            0,2,0,0, 0,0,0,0, 2, 0,
            1,0,0,0, -1,0,0,-1, 0, 1;
    Matrix<double, 10, 10> A3 = QR.transpose() * A2 * QR;
    VectorXd data(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            data(10*i+j) = A3(j, i);
        }
    }

    MatrixXcd a = solver_opt_pnp_hesch2_red(data);
    sols_p->clear();
    for (int i = 0; i < a.cols(); i++)
    {
        if (fabs(a.col(i)(0).imag()) == 0)
        {
            Vector3d s = a.col(i).real();
            Matrix3d Rc = (1-s.transpose()*s)*Matrix3d::Identity()+2*cpmat(s)+2*s*s.transpose();
            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            VectorXd rvec(9);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            Vector3d tc = -T.colPivHouseholderQr().solve( A * rvec);
            Matrix4d Tc;
            Tc.setIdentity();
            Tc.block<3,3>(0,0) = Rc;
            Tc.block<3,1>(0,3) = tc;
            sols_p->push_back(Tc);
        }
    }
}

void ProjectionJacobian(const Vector3d& X, Matrix<double, 2, 3>* J_p)
{
    Matrix<double, 2, 3> J;
    J.setZero();
    J(0, 0) = 1.0/X(2);
    J(0, 2) = -X(0)/X(2)/X(2);
    J(1, 1) = 1.0/X(2);
    J(1, 2) = -X(1)/X(2)/X(2);
    *J_p = J;
}

void Unpack(const Eigen::Vector3d& r_vec, const Eigen::Vector3d t_vec, VectorXd* sol_p)
{
    VectorXd sol(12);
    cv::Mat r_vec_cv;
    cv::eigen2cv(r_vec, r_vec_cv);
    cv::Mat R_cv;
    cv::Rodrigues(r_vec_cv, R_cv);
    Matrix3d R;
    cv::cv2eigen(R_cv, R);
    for (int i = 0; i < 3; i++)
    {
        sol.segment<3>(3*i) = R.row(i).transpose();
    }
    sol.segment<3>(9) = t_vec;
    *sol_p = sol;
}

void MLPnP(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
          const std::vector<float>& sigmas2d_norm,
          posevector* sols_p)
{
    int n = xx.size();
    Eigen::MatrixXd A(2*n, 12);
    mat2dvector pis;
    for (int i = 0; i < xx.size(); i++)
    {
        Matrix<double, 3, 2> j_pi;
        j_pi.setZero();
        j_pi.block<2, 2>(0, 0).setIdentity();
        Matrix3d sigma_xx = j_pi * sigmas2d_norm[i] * Matrix2d::Identity() * j_pi.transpose();
        Vector3d v;
        v.segment<2>(0) = xx[i];
        v(2) = 1.0;
        v = v / v.norm();
        Matrix3d vtv = v * v.transpose();
        Matrix3d j = (Matrix3d::Identity() - vtv) * 1.0/xx[i].norm();
        Matrix3d sigma_vv = j * sigma_xx * j.transpose();
        JacobiSVD<Matrix<double, 1, 3>> svd(v.transpose(), ComputeFullU | ComputeFullV);
        Vector3d r = svd.matrixV().col(2);
        Vector3d s = svd.matrixV().col(1);
//        std::cout << " check svd: " << r.dot(v) << " " << s.dot(v) << std::endl;
        Matrix<double, 3, 2> j_vr = svd.matrixV().block<3, 2>(0, 1);
        Matrix2d sigma_vr = j_vr.transpose() * (sigma_vv * j_vr);
        A.block<1, 3>(2*i, 9) = r.transpose();
        A.block<1, 3>(2*i+1, 9) = s.transpose();
        Matrix<double, 3, 9> r_x;
        r_x.setZero();
        r_x.block<1, 3>(0, 0) = XX[i].transpose();
        r_x.block<1, 3>(1, 3) = XX[i].transpose();
        r_x.block<1, 3>(2, 6) = XX[i].transpose();
        Matrix< double, 2, 3> J_ort;
        J_ort.row(0) = r.transpose();
        J_ort.row(1) = s.transpose();
        A.block<2, 9>(2*i, 0) = J_ort * r_x;



        Matrix2d Pi = sigma_vr.inverse();
        pis.push_back(Pi);
    }
    MatrixXd PA(2*n, 12);
    for (int i = 0; i < xx.size(); i++)
    {
//        JacobiSVD<Matrix2d> svd_2d(pis[i], ComputeFullU | ComputeFullV);
//        Vector2d s = svd_2d.singularValues();
//        s[0] = sqrt(s[0]);
//        s[1] = sqrt(s[1]);
//        DiagonalMatrix<double, 2> s_sq = s.asDiagonal();
//        Matrix2d pi_sq = svd_2d.matrixU() * s_sq * svd_2d.matrixV().transpose();
        PA.block<2, 12>(2*i, 0) = pis[i] * A.block<2, 12>(2*i, 0);
    }

    Matrix<double, 12, 12> N = A.transpose() * PA;
    JacobiSVD<Matrix<double, 12, 12>> svd(N, ComputeFullU | ComputeFullV);
    VectorXd s = svd.singularValues();
    VectorXd s_sqrt (12);
    for (int  i = 0; i < 12; i++)
    {
        s_sqrt[i] = sqrt(fabs(s[i]));
    }

    MatrixXd N_sqrt = s_sqrt.asDiagonal() * svd.matrixV().transpose();

    VectorXd sol = svd.matrixV().col(11);
    Vector3d t_dir = sol.segment<3>(9);
    Matrix3d R_est;
    R_est.row(0) = sol.segment<3>(0).transpose();
    R_est.row(1) = sol.segment<3>(3).transpose();
    R_est.row(2) = sol.segment<3>(6).transpose();
    t_dir = t_dir / pow(R_est.row(0).norm() * R_est.row(1).norm() * R_est.row(2).norm(), 1.0/3.0);
    JacobiSVD<Matrix3d> r_svd(R_est, ComputeFullU | ComputeFullV);
    Matrix3d R = r_svd.matrixU() * r_svd.matrixV().transpose();
    //next: nonlinear refinement

    cv::Mat R_cv;
    cv::eigen2cv(R, R_cv);
    cv::Mat r_vec_cv;
    cv::Rodrigues(R_cv, r_vec_cv);

    for (int i = 0; i < 3; i++)
    {
        sol.segment<3>(3*i) = R.row(i).transpose();
    }
    sol.segment<3>(9) = t_dir;

    double err_norm = sol.dot(N * sol);
    double err_norm_2 = (N_sqrt * sol).dot(N_sqrt * sol);

    Vector3d r_vec;
    cv::cv2eigen(r_vec_cv, r_vec);
    Vector3d t_eig = t_dir;

    Vector3d r_sol = r_vec;
    Vector3d t_sol = t_eig;

    std::cout << " init err norm " << err_norm << " " << err_norm_2 << std::endl;

    for (int it = 0; it < 5; it++)
    {
        cv::Mat jac_cv;
        cv::eigen2cv(r_sol, r_vec_cv);
        cv::Rodrigues(r_vec_cv, R_cv, jac_cv);
        //assuming jac_cv = 3x9
        Matrix<double, 3, 9> jac;
        cv::cv2eigen(jac_cv, jac);
        Eigen::Matrix<double, 12, 6> J_param;
        J_param.setZero();
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                J_param.block<1, 3>(3*i+j, 0) = jac.col(3*j + i).transpose();
            }
        }
        J_param.block<9, 3>(0, 0) = jac.transpose();
        J_param.block<3, 3>(9, 3).setIdentity();
        Matrix<double, 12, 6> J = N_sqrt * J_param;

        VectorXd x(6);
        x.segment<3>(0) = r_sol;
        x.segment<3>(3) = t_sol;
        Unpack(r_sol, t_sol, &sol);
        x = x - (J.transpose() * J).inverse() * J.transpose() * N_sqrt * sol;

        r_vec = x.segment<3>(0);
        t_eig = x.segment<3>(3);
        Unpack(r_vec, t_eig, &sol);

        double err_norm_curr = sol.dot(N * sol);

        std::cout << " ref err norm " << err_norm_curr << std::endl;

        if (err_norm_curr > err_norm)
        {
//            std::cout << " did " << it << " its " << std::endl;
            break;
        } else {
            err_norm = err_norm_curr;
            r_sol = r_vec;
            t_sol = t_eig;
        }
    }


    Matrix4d T;

    cv::eigen2cv(r_sol, r_vec_cv);
    cv::Rodrigues(r_vec_cv, R_cv);
    cv::cv2eigen(R_cv, R);

    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t_sol;
    T(3, 3) = 1.0;
    sols_p->push_back(T);
}

void DLSU(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
          const std::vector<float>& sigmas3d,
          const std::vector<float>& sigmas2d_norm,
          const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
          bool is_use_3d, bool is_use_2d)
{
    DLSULines(XX, xx,
            std::vector<Eigen::Vector3d>(), std::vector<Eigen::Vector3d>(),
            std::vector<Eigen::Vector3d>(),
            sigmas3d,
            sigmas2d_norm,
            mat6dvector(),
            std::vector<float>(),
            R_est, t_est,
            sols_p,
            is_use_3d, is_use_2d);
}

void DLSULines(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx_n,
               const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
               const std::vector<Eigen::Vector3d>& l2ds,
               const std::vector<float>& sigmas3d,
               const std::vector<float>& sigmas2d_norm,
               const mat6dvector& sigmas3d_lines,
               const std::vector<float>& sigmasDetLines,
               const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est,
               posevector* sols_p,
               bool is_use_3d, bool is_use_2d)
{
    int np = XX.size();
    std::vector<double> depths_est;

    std::vector<Eigen::Matrix<double, 2, 3>> Tps;
    std::vector<Eigen::Matrix<double, 2, 3>> Tls;
    Eigen::Matrix3d T;
    T.setZero();
    Eigen::Matrix<double, 3, 9> A;
    A.setZero();

    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api, Ali;
    for (int i = 0; i < np; i++)
    {
//        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
//        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
        const Eigen::Vector2d& x = xx_n[i];
        Eigen::Matrix2d Sigma2;
        Sigma2.setZero();
        Eigen::Matrix2d Sigma2_2d;

        if (is_use_3d) {
            float s3d = sigmas3d[i];
            Sigma2(0, 0) = s3d * (x(0) * x(0) + 1);
            Sigma2(1, 1) = s3d * (x(1) * x(1) + 1);
            Sigma2(0, 1) = s3d * x(0) * x(1);
            Sigma2(1, 0) = Sigma2(0, 1);
            Sigma2_2d.setZero();
        } else {
            Sigma2_2d.setIdentity();
        }
        //Sigma2 = Sigma2 + (depths_est[i]*depths_est[i]*sigmas2d_norm[i] + 1e-6)*Eigen::Matrix2d::Identity();

        if (is_use_2d)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            double d = Xc(2);
            Sigma2_2d = (d*d*sigmas2d_norm[i] + 1e-6)*Eigen::Matrix2d::Identity();
        }
        Sigma2 = Sigma2 + Sigma2_2d;

        Eigen::Matrix2d Sigma2Inv = Sigma2.inverse();
        S2s.push_back(Sigma2Inv);
        Eigen::Matrix<double,2,3> Ti;
        Ti << -1,0, x(0),
                0, -1, x(1);
        Tps.push_back(Ti);
        T = T + Ti.transpose() * Sigma2Inv*Ti;
        const Eigen::Vector3d& Xp = XX[i];
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0, 0) = -Xp.transpose();
        Ai.block<1,3>(0, 6) = x(0)*Xp.transpose();
        Ai.block<1,3>(1, 3) = -Xp.transpose();
        Ai.block<1,3>(1, 6) = x(1)*Xp.transpose();
        Api.push_back(Ai);
        A = A + Ti.transpose()*Sigma2Inv*Ai;
    }


    int nl = XXs.size();
    S2ls.resize(nl);
    Tls.resize(nl);
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix2d Sigma2;
        const Eigen::Matrix<double,6,6>& S6D = sigmas3d_lines[i];
        Sigma2(0,0) = l2ds[i].transpose() * R_est * S6D.block<3,3>(0,0) * R_est.transpose() * l2ds[i];
        Sigma2(0,1) = l2ds[i].transpose() * R_est * S6D.block<3,3>(0,3) * R_est.transpose() * l2ds[i];
        Sigma2(1,1) = l2ds[i].transpose() * R_est * S6D.block<3,3>(3,3) * R_est.transpose() * l2ds[i];
        Sigma2(1,0) = l2ds[i].transpose() * R_est * S6D.block<3,3>(3,0) * R_est.transpose() * l2ds[i];
        Eigen::Matrix2d Sigma2_2d;
        Eigen::Vector3d Xsc = R_est*XXs[i]+t_est;
        Eigen::Vector3d Xec = R_est*XXe[i]+t_est;
//        const Eigen::Matrix3d& s2d = sigmas2d_lines[i];

//        Sigma2_2d.setIdentity();
//        Sigma2_2d(0,0) = Xsc(2)*Xsc(2);
//        Sigma2_2d(1,1) = Xec(2)*Xec(2);
//        Sigma2_2d *= sigmasDetLines[i]/K(0,0)/K(0,0);

//        Sigma2_2d(0,0) = Xsc.transpose() * s2d * Xsc;
//        Sigma2_2d(1,1) = Xec.transpose() * s2d * Xec;
//        Sigma2_2d(0,1) = Xsc.transpose() * s2d * Xec;
//        Sigma2_2d(1,0) = Xec.transpose() * s2d * Xsc;
        Sigma2_2d = Eigen::Matrix2d::Identity() * sigmasDetLines[i];
        Sigma2_2d(0,0) = Sigma2_2d(0,0) * Xsc(2)*Xsc(2);
        Sigma2_2d(1,1) = Sigma2_2d(1,1) * Xec(2)*Xec(2);

        Sigma2 = Sigma2_2d + Sigma2;
        S2ls[i] = (Sigma2 + 1e-8*Eigen::Matrix2d::Identity()).inverse();

        Eigen::Matrix<double,2,3> Ti;
        Ti.block<1,3>(0,0) = l2ds[i].transpose();
        Ti.block<1,3>(1,0) = l2ds[i].transpose();
        Eigen::Matrix<double,3,9> Xsi;
        Xsi.setZero();
        const Eigen::Vector3d& Xs = XXs[i];
        Xsi.block<1,3>(0,0) = Xs.transpose();
        Xsi.block<1,3>(1,3) = Xs.transpose();
        Xsi.block<1,3>(2,6) = Xs.transpose();
        Eigen::Matrix<double,1,9> Asi = l2ds[i].transpose() * Xsi;
        Eigen::Matrix<double,3,9> Xei;
        Xei.setZero();
        const Eigen::Vector3d& Xe = XXe[i];
        Xei.block<1,3>(0,0) = Xe.transpose();
        Xei.block<1,3>(1,3) = Xe.transpose();
        Xei.block<1,3>(2,6) = Xe.transpose();
        Eigen::Matrix<double, 1, 9> Aei = l2ds[i].transpose() * Xei;
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.block<1,9>(0,0) = Asi;
        Ai.block<1,9>(1,0) = Aei;
        T = T + Ti.transpose() * S2ls[i] * Ti;
        A = A + Ti.transpose() * S2ls[i] * Ai;
        Ali.push_back(Ai);
        Tls[i] = Ti;
    }
//    std::vector<float> s2ls;
//    std::vector<Eigen::Matrix<double,1,9>> alis;
//    for (int i = 0; i < nl; i++)
//    {
//        s2ls.push_back(1.0/(sigmasDetLines[i] + l2ds[i].transpose()*(R_est*sigmas3d_lines[i].block<3,3>(3,3)*R_est.transpose())*l2ds[i]));
//        Eigen::Matrix<double,3,9> Xi;
//        Eigen::Matrix<double, 3, 9> Xsi;
//        Xsi.setZero();
//        const Eigen::Vector3d& Xs = XXe[i]-XXs[i];
//        Xsi.block<1,3>(0,0) = Xs.transpose();
//        Xsi.block<1,3>(1,3) = Xs.transpose();
//        Xsi.block<1,3>(2,6) = Xs.transpose();
//        alis.push_back(l2ds[i].transpose() * Xsi);
//    }


    Eigen::Matrix<double, 9, 9> A2;
    A2.setZero();

    Eigen::Matrix<double, 3,9> T_express = T.colPivHouseholderQr().solve(-A);
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Api[i] + Tps[i]*T_express;
        A2 = A2 + Ai.transpose() * S2s[i] * Ai;
    }
//    for i = 1:nl
//    Ai = Ali(:, :, i)-Tl(:,:,i)*(T\A1);
//    A2 = A2 + Ai'*S2ls(:, :, i)*Ai;
//    end
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Ali[i] + Tls[i] * T_express;
        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
    }

//        for (int i = 0; i < nl; i++)
//    {
//        A2 = A2 + alis[i].transpose() * s2ls[i] * alis[i];
//    }


//    for (int i = 0; i < nl; i++)
//    {
//        Eigen::Matrix<double, 2, 9> Ai = Ali[i]+Tli[i]*T_express;
//        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
//    }

    Matrix<double, 9, 10> QR;
    QR << 1,0,0,0, 1,0,0,-1, 0, -1,
            0,0,0,-2, 0,2,0,0, 0, 0,
            0,0,2,0, 0,0,2,0, 0, 0,
            0,0,0,2, 0,2,0,0, 0, 0,
            1,0,0,0, -1,0,0,1, 0, -1,
            0,-2,0,0, 0,0,0,0, 2, 0,
            0,0,-2,0, 0,0,2,0, 0, 0,
            0,2,0,0, 0,0,0,0, 2, 0,
            1,0,0,0, -1,0,0,-1, 0, 1;
    Matrix<double, 10, 10> A3 = QR.transpose() * A2 * QR;
    VectorXd data(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            data(10*i+j) = A3(j, i);
        }
    }

    MatrixXcd a = solver_opt_pnp_hesch2_red(data);
    sols_p->clear();
    for (int i = 0; i < a.cols(); i++)
    {
        if (fabs(a.col(i)(0).imag()) == 0)
        {
            Vector3d s = a.col(i).real();
            Matrix3d Rc = (1-s.transpose()*s)*Matrix3d::Identity()+2*cpmat(s)+2*s*s.transpose();
            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            VectorXd rvec(9);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            Vector3d tc = -T.colPivHouseholderQr().solve( A * rvec);
            bool is_behind = false;
            for (int pti = 0; pti < XX.size(); pti++)
            {
                Eigen::Vector3d Xc = Rc*XX[pti] + tc;
                if (Xc(2)<0)
                {
                    is_behind = true;
                }
            }
            if (is_behind)
                continue;


            Eigen::Vector3d sp = -s;
            for (int it = 0; it < 1; it++)
            {
                double g[3];
//                void gradient(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* g);
//                void hessian(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* H);
                double H[9];
                gradient(sp(0), sp(1), sp(2), A2, g);
                hessian(sp(0), sp(1), sp(2), A2, H);
                Map<Eigen::Matrix<double, 3, 1> > g_eig(g);
                Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > H_eig(H);
//                std::cout << "hessian " << H_eig << std::endl;
//                std::cout << "gradient " << g_eig << std::endl;
                sp = sp - H_eig.inverse() * g_eig;
            }
//            std::cout << " refined was " << s << " is " << -sp << std::endl;
            Eigen::Vector3d sfin = -sp;
            Rc = 1.0/(1+sfin.transpose()*sfin)*((1-sfin.transpose()*sfin)*Matrix3d::Identity()+2*cpmat(sfin)+2*sfin*sfin.transpose());

//            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            tc = -T.colPivHouseholderQr().solve( A * rvec);
            Matrix4d Tc;
            Tc.setIdentity();
            Tc.block<3,3>(0,0) = Rc;
            Tc.block<3,1>(0,3) = tc;
            sols_p->push_back(Tc);
        }
    }
}

void DLSLines(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
              const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
              const std::vector<Eigen::Vector3d>& l2ds,
              const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p)
{
    int np = XX.size();
    std::vector<double> depths_est;

    std::vector<Eigen::Matrix<double, 2, 3>> Tps;
    std::vector<Eigen::Matrix<double, 2, 3>> Tls;
    Eigen::Matrix3d T;
    T.setZero();
    Eigen::Matrix<double, 3, 9> A;
    A.setZero();

    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;
    for (int i = 0; i < np; i++)
    {
//        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
//        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
        const Eigen::Vector2d& x = xx[i];
        S2s.push_back(Eigen::Matrix2d::Identity());
        Eigen::Matrix<double,2,3> Ti;
        Ti << -1,0, x(0),
                0, -1, x(1);
        Tps.push_back(Ti);
        T = T + Ti.transpose() * Ti;
        const Eigen::Vector3d& Xp = XX[i];
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0, 0) = -Xp.transpose();
        Ai.block<1,3>(0, 6) = x(0)*Xp.transpose();
        Ai.block<1,3>(1, 3) = -Xp.transpose();
        Ai.block<1,3>(1, 6) = x(1)*Xp.transpose();
        Api.push_back(Ai);
        A = A + Ti.transpose()*Ai;
    }

    int nl = XXs.size();
    std::vector<Eigen::Matrix<double, 2, 9>> Ali;
    std::vector<Eigen::Matrix<double, 2, 3>> Tli;
    for (int i = 0; i < nl; i++)
    {
        const Eigen::Vector3d& lvec = l2ds[i];
        S2ls.push_back(Eigen::Matrix2d::Identity());
        Eigen::Matrix<double, 2, 3> Ti;
        Ti.row(0) = lvec.transpose();
        Ti.row(1) = lvec.transpose();
        Tli.push_back(Ti);
        const Eigen::Vector3d& Xsi = XXs[i];
        Eigen::Matrix<double, 3, 9> Xsmat;
        Xsmat.setZero();
        Xsmat.block<1,3>(0,0) = Xsi.transpose();
        Xsmat.block<1,3>(1,3) = Xsi.transpose();
        Xsmat.block<1,3>(2,6) = Xsi.transpose();
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.row(0) = lvec.transpose() * Xsmat;
        const Eigen::Vector3d& Xei = XXe[i];
        Xsmat.block<1,3>(0,0) = Xei.transpose();
        Xsmat.block<1,3>(1,3) = Xei.transpose();
        Xsmat.block<1,3>(2,6) = Xei.transpose();
        Ai.row(1) = lvec.transpose() * Xsmat;
        Ali.push_back(Ai);
        Eigen::Matrix<double, 3, 3> Tl = Ti.transpose()*Ti;
        T = T + Tl;
        Tls.push_back(Ti);
        A = A + Ti.transpose() * Ai;
    }

//
//    S2ls.resize(nl);
//    Tls.resize(nl);
//    for (int i = 0; i < nl; i++)
//    {
////        std::cout << " dlsl l2d " << l2ds[i] << std::endl;
//        S2ls[i] = Eigen::Matrix2d::Identity();
//
//        Eigen::Matrix<double,2,3> Ti;
//        Ti.block<1,3>(0,0) = l2ds[i].transpose();
//        Ti.block<1,3>(1,0) = l2ds[i].transpose();
//        Eigen::Matrix<double,3,9> Xsi;
//        Xsi.setZero();
//        const Eigen::Vector3d& Xs = XXs[i];
//        Xsi.block<1,3>(0,0) = Xs.transpose();
//        Xsi.block<1,3>(1,3) = Xs.transpose();
//        Xsi.block<1,3>(2,6) = Xs.transpose();
//        Eigen::Matrix<double,1,9> Asi = l2ds[i].transpose() * Xsi;
//        Eigen::Matrix<double,3,9> Xei;
//        Xei.setZero();
//        const Eigen::Vector3d& Xe = XXe[i];
//        Xei.block<1,3>(0,0) = Xe.transpose();
//        Xei.block<1,3>(1,3) = Xe.transpose();
//        Xei.block<1,3>(2,6) = Xe.transpose();
//        Eigen::Matrix<double, 1, 9> Aei = l2ds[i].transpose() * Xei;
//        Eigen::Matrix<double, 2, 9> Ai;
//        Ai.block<1,9>(0,0) = Asi;
//        Ai.block<1,9>(1,0) = Aei;
//        T = T + Ti.transpose() * S2ls[i] * Ti;
//        A = A + Ti.transpose() * S2ls[i] * Ai;
//        Ali.push_back(Ai);
//        Tls[i] = Ti;
//    }
//    std::vector<float> s2ls;
//    std::vector<Eigen::Matrix<double,1,9>> alis;
//    for (int i = 0; i < nl; i++)
//    {
//        s2ls.push_back(1.0/(sigmasDetLines[i] + l2ds[i].transpose()*(R_est*sigmas3d_lines[i].block<3,3>(3,3)*R_est.transpose())*l2ds[i]));
//        Eigen::Matrix<double,3,9> Xi;
//        Eigen::Matrix<double, 3, 9> Xsi;
//        Xsi.setZero();
//        const Eigen::Vector3d& Xs = XXe[i]-XXs[i];
//        Xsi.block<1,3>(0,0) = Xs.transpose();
//        Xsi.block<1,3>(1,3) = Xs.transpose();
//        Xsi.block<1,3>(2,6) = Xs.transpose();
//        alis.push_back(l2ds[i].transpose() * Xsi);
//    }


    Eigen::Matrix<double, 9, 9> A2;
    A2.setZero();

    Eigen::Matrix<double, 3,9> T_express = T.colPivHouseholderQr().solve(-A);
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Api[i] + Tps[i]*T_express;
        A2 = A2 + Ai.transpose() * S2s[i] * Ai;
    }
//    for i = 1:nl
//    Ai = Ali(:, :, i)-Tl(:,:,i)*(T\A1);
//    A2 = A2 + Ai'*S2ls(:, :, i)*Ai;
//    end
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Ali[i] + Tls[i]*T_express;
        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
    }


//        for (int i = 0; i < nl; i++)
//    {
//        A2 = A2 + alis[i].transpose() * s2ls[i] * alis[i];
//    }


//    for (int i = 0; i < nl; i++)
//    {
//        Eigen::Matrix<double, 2, 9> Ai = Ali[i]+Tli[i]*T_express;
//        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
//    }

    Matrix<double, 9, 10> QR;
    QR << 1,0,0,0, 1,0,0,-1, 0, -1,
            0,0,0,-2, 0,2,0,0, 0, 0,
            0,0,2,0, 0,0,2,0, 0, 0,
            0,0,0,2, 0,2,0,0, 0, 0,
            1,0,0,0, -1,0,0,1, 0, -1,
            0,-2,0,0, 0,0,0,0, 2, 0,
            0,0,-2,0, 0,0,2,0, 0, 0,
            0,2,0,0, 0,0,0,0, 2, 0,
            1,0,0,0, -1,0,0,-1, 0, 1;
    Matrix<double, 10, 10> A3 = QR.transpose() * A2 * QR;
    VectorXd data(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            data(10*i+j) = A3(j, i);
        }
    }

    MatrixXcd a = solver_opt_pnp_hesch2_red(data);
    sols_p->clear();
    for (int i = 0; i < a.cols(); i++)
    {
        if (fabs(a.col(i)(0).imag()) == 0)
        {
            Vector3d s = a.col(i).real();
            Matrix3d Rc = (1-s.transpose()*s)*Matrix3d::Identity()+2*cpmat(s)+2*s*s.transpose();

            Eigen::Vector3d sp = -s;
            for (int it = 0; it < 1; it++)
            {
                double g[3];
//                void gradient(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* g);
//                void hessian(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* H);
                double H[9];
                gradient(sp(0), sp(1), sp(2), A2, g);
                hessian(sp(0), sp(1), sp(2), A2, H);
                Map<Eigen::Matrix<double, 3, 1> > g_eig(g);
                Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > H_eig(H);
//                std::cout << "hessian " << H_eig << std::endl;
//                std::cout << "gradient " << g_eig << std::endl;
                sp = sp - H_eig.inverse() * g_eig;
            }
//            std::cout << " refined was " << s << " is " << -sp << std::endl;
            Eigen::Vector3d sfin = -sp;
            Rc = 1.0/(1+sfin.transpose()*sfin)*((1-sfin.transpose()*sfin)*Matrix3d::Identity()+2*cpmat(sfin)+2*sfin*sfin.transpose());

//            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            VectorXd rvec(9);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            Vector3d tc = -T.colPivHouseholderQr().solve( A * rvec);
            Matrix4d Tc;
            Tc.setIdentity();
            Tc.block<3,3>(0,0) = Rc;
            Tc.block<3,1>(0,3) = tc;
            sols_p->push_back(Tc);
        }
    }
}


void DLSULines_accurate(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx_n,
                        const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
                        const std::vector<Eigen::Vector3d>& l2ds,
                        const std::vector<Eigen::Matrix3d>& sigmas3d,
                        const std::vector<float>& sigmas2d_norm,
                        const mat6dvector& sigmas3d_lines,
                        const std::vector<Eigen::Matrix3d>& sigmas2d_lines,
                        const Eigen::Matrix3d& R_est,
                        const Eigen::Vector3d& t_est, posevector* sols_p)
{
    int np = XX.size();
    std::vector<double> depths_est;

    std::vector<Eigen::Matrix<double, 2, 3>> Tps;
    std::vector<Eigen::Matrix<double, 2, 3>> Tls;
    Eigen::Matrix3d T;
    T.setZero();
    Eigen::Matrix<double, 3, 9> A;
    A.setZero();
    bool is_est_available = false;
//    if (t_est.norm()>0)
//    {
        for (int i = 0; i < np; i++)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
        }
        is_est_available = true;
//    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix2d Sigma2;
//        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
//        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
        const Eigen::Vector2d& x = xx_n[i];
        Eigen::Matrix3d S3d = R_est*sigmas3d[i]*R_est.transpose();
        Sigma2(0,0) = S3d(2,2)*x(0)*x(0)- 2*S3d(0,2)*x(0)+S3d(0,0);
        Sigma2(1,1) = S3d(2,2)*x(1)*x(1)- 2*S3d(1,2)*x(1)+S3d(1,1);
        Sigma2(0,1) = S3d(2,2)*x(0)*x(1)- S3d(0,2)*x(1)-S3d(1,2)*x(0)+S3d(0,1);
        Sigma2(1,0) = Sigma2(0,1);

        Sigma2 = Sigma2 + (depths_est[i]*depths_est[i]*sigmas2d_norm[i] + 1e-6)*Eigen::Matrix2d::Identity();

        Eigen::Matrix2d Sigma2Inv = Sigma2.inverse();
        S2s.push_back(Sigma2Inv);
        Eigen::Matrix<double,2,3> Ti;
        Ti << -1,0, x(0),
                0, -1, x(1);
        Tps.push_back(Ti);
        T = T + Ti.transpose() * Sigma2Inv*Ti;
        const Eigen::Vector3d& Xp = XX[i];
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0, 0) = -Xp.transpose();
        Ai.block<1,3>(0, 6) = x(0)*Xp.transpose();
        Ai.block<1,3>(1, 3) = -Xp.transpose();
        Ai.block<1,3>(1, 6) = x(1)*Xp.transpose();
        Api.push_back(Ai);
        A = A + Ti.transpose()*Sigma2Inv*Ai;
    }
    std::vector<Eigen::Matrix<double, 2, 9>> Ali;
    int nl = XXs.size();
    S2ls.resize(nl);
    Tls.resize(nl);
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix2d Sigma2;
        const Eigen::Matrix<double,6,6>& S6D = sigmas3d_lines[i];
        Sigma2(0,0) = l2ds[i].transpose() * R_est * S6D.block<3,3>(0,0) * R_est.transpose() * l2ds[i];
        Sigma2(0,1) = l2ds[i].transpose() * R_est * S6D.block<3,3>(0,3) * R_est.transpose() * l2ds[i];
        Sigma2(1,1) = l2ds[i].transpose() * R_est * S6D.block<3,3>(3,3) * R_est.transpose() * l2ds[i];
        Sigma2(1,0) = l2ds[i].transpose() * R_est * S6D.block<3,3>(3,0) * R_est.transpose() * l2ds[i];
        Eigen::Matrix2d Sigma2_2d;
        Eigen::Vector3d Xsc = R_est*XXs[i]+t_est;
        Eigen::Vector3d Xec = R_est*XXe[i]+t_est;
        const Eigen::Matrix3d& s2d = sigmas2d_lines[i];
        Sigma2_2d(0,0) = Xsc.transpose()*s2d*Xsc;
        Sigma2_2d(1,1) = Xec.transpose()*s2d*Xec;
        Sigma2_2d(0,1) = Xsc.transpose()*s2d*Xec;
        Sigma2_2d(1,0) = Xec.transpose()*s2d*Xsc;
//        Sigma2_2d.setIdentity();
//        Sigma2_2d(0,0) = Xsc(2)*Xsc(2);
//        Sigma2_2d(1,1) = Xec(2)*Xec(2);
//        Sigma2_2d *= sigmasDetLines[i]/K(0,0)/K(0,0);
        Sigma2 = Sigma2 + Sigma2_2d;
        S2ls[i] = (Sigma2 + 1e-8*Eigen::Matrix2d::Identity()).inverse();
//        Ti = [l(:, i)'; l(:, i)'];
//        Xsi = Xs(:, i);
//        Asi = l(:, i)'*[Xsi', zeros(1,6);
//        zeros(1,3), Xsi', zeros(1,3);
//        zeros(1,6), Xsi'];
//        Xei = Xe(:, i);
//        Aei = l(:, i)'*[Xei', zeros(1,6);
//        zeros(1,3), Xei', zeros(1,3);
//        zeros(1,6), Xei'];
//        Ai = [Asi; Aei];
//        Ali(:, :, i) = Ai;
//        T = T + Ti'*S2ls(:,:,i)*Ti;
//        A = A + Ti'*S2ls(:, :, i)*Ai;
//        Tl(:, :, i) = Ti;
        Eigen::Matrix<double,2,3> Ti;
        Ti.block<1,3>(0,0) = l2ds[i].transpose();
        Ti.block<1,3>(1,0) = l2ds[i].transpose();
        Eigen::Matrix<double,3,9> Xsi;
        Xsi.setZero();
        const Eigen::Vector3d& Xs = XXs[i];
        Xsi.block<1,3>(0,0) = Xs.transpose();
        Xsi.block<1,3>(1,3) = Xs.transpose();
        Xsi.block<1,3>(2,6) = Xs.transpose();
        Eigen::Matrix<double,1,9> Asi = l2ds[i].transpose() * Xsi;
        Eigen::Matrix<double,3,9> Xei;
        Xei.setZero();
        const Eigen::Vector3d& Xe = XXe[i];
        Xei.block<1,3>(0,0) = Xe.transpose();
        Xei.block<1,3>(1,3) = Xe.transpose();
        Xei.block<1,3>(2,6) = Xe.transpose();
        Eigen::Matrix<double, 1, 9> Aei = l2ds[i].transpose() * Xei;
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.block<1,9>(0,0) = Asi;
        Ai.block<1,9>(1,0) = Aei;
        T = T + Ti.transpose() * S2ls[i] * Ti;
        A = A + Ti.transpose() * S2ls[i] * Ai;
        Ali.push_back(Ai);
        Tls[i] = Ti;
    }

//    std::vector<float> s2ls;
//    std::vector<Eigen::Matrix<double,1,9>> alis;
//    for (int i = 0; i < nl; i++)
//    {
//        s2ls.push_back(1.0/(sigmasDetLines[i] + l2ds[i].transpose()*(R_est*sigmas3d_lines[i].block<3,3>(3,3)*R_est.transpose())*l2ds[i]));
//        Eigen::Matrix<double,3,9> Xi;
//        Eigen::Matrix<double, 3, 9> Xsi;
//        Xsi.setZero();
//        const Eigen::Vector3d& Xs = XXe[i]-XXs[i];
//        Xsi.block<1,3>(0,0) = Xs.transpose();
//        Xsi.block<1,3>(1,3) = Xs.transpose();
//        Xsi.block<1,3>(2,6) = Xs.transpose();
//        alis.push_back(l2ds[i].transpose() * Xsi);
//    }


    Eigen::Matrix<double, 9, 9> A2;
    A2.setZero();

    Eigen::Matrix<double, 3,9> T_express = T.colPivHouseholderQr().solve(-A);
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Api[i] + Tps[i]*T_express;
        A2 = A2 + Ai.transpose() * S2s[i] * Ai;
    }

//    for (int i = 0; i < nl; i++)
//    {
//        A2 = A2 + alis[i].transpose() * s2ls[i] * alis[i];
//    }


    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Ali[i] + Tls[i]*T_express;
        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
    }
//

    Matrix<double, 9, 10> QR;
    QR << 1,0,0,0, 1,0,0,-1, 0, -1,
            0,0,0,-2, 0,2,0,0, 0, 0,
            0,0,2,0, 0,0,2,0, 0, 0,
            0,0,0,2, 0,2,0,0, 0, 0,
            1,0,0,0, -1,0,0,1, 0, -1,
            0,-2,0,0, 0,0,0,0, 2, 0,
            0,0,-2,0, 0,0,2,0, 0, 0,
            0,2,0,0, 0,0,0,0, 2, 0,
            1,0,0,0, -1,0,0,-1, 0, 1;
    Matrix<double, 10, 10> A3 = QR.transpose() * A2 * QR;
    VectorXd data(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            data(10*i+j) = A3(j, i);
        }
    }

    MatrixXcd a = solver_opt_pnp_hesch2_red(data);
    sols_p->clear();
    for (int i = 0; i < a.cols(); i++)
    {
        if (fabs(a.col(i)(0).imag()) == 0)
        {
            Vector3d s = a.col(i).real();
            Matrix3d Rc = (1-s.transpose()*s)*Matrix3d::Identity()+2*cpmat(s)+2*s*s.transpose();

            Eigen::Vector3d sp = -s;
            for (int it = 0; it < 1; it++)
            {
                double g[3];
//                void gradient(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* g);
//                void hessian(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* H);
                double H[9];
                gradient(sp(0), sp(1), sp(2), A2, g);
                hessian(sp(0), sp(1), sp(2), A2, H);
                Map<Eigen::Matrix<double, 3, 1> > g_eig(g);
                Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > H_eig(H);
//                std::cout << "hessian " << H_eig << std::endl;
//                std::cout << "gradient " << g_eig << std::endl;
                sp = sp - H_eig.inverse() * g_eig;
            }
//            std::cout << " refined was " << s << " is " << -sp << std::endl;
            Eigen::Vector3d sfin = -sp;
            Rc = 1.0/(1+sfin.transpose()*sfin)*((1-sfin.transpose()*sfin)*Matrix3d::Identity()+2*cpmat(sfin)+2*sfin*sfin.transpose());

//            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            VectorXd rvec(9);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            Vector3d tc = -T.colPivHouseholderQr().solve( A * rvec);
            Matrix4d Tc;
            Tc.setIdentity();
            Tc.block<3,3>(0,0) = Rc;
            Tc.block<3,1>(0,3) = tc;
            sols_p->push_back(Tc);
        }
    }
}

void DLSU_accurate(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
          const std::vector<Eigen::Matrix3d>& sigmas3d,
          const std::vector<float>& sigmas2d_norm,
          const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
          bool is_use_3d, bool is_use_2d)
{
    int np = XX.size();
    std::vector<double> depths_est;

    std::vector<Eigen::Matrix<double, 2, 3>> Tps;
    std::vector<Eigen::Matrix<double, 3, 3>> Tls;
    Eigen::Matrix3d T;
    T.setZero();
    Eigen::Matrix<double, 3, 9> A;
    A.setZero();
    bool is_est_available = false;
//    if (t_est.norm()>0)
//    {
//        for (int i = 0; i < np; i++)
//        {
//            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
//            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
//        }
//        is_est_available = true;
//    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;
    for (int i = 0; i < np; i++)
    {

//        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
//        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
        const Eigen::Vector2d& x = xx[i];
        Eigen::Matrix2d Sigma2;
        Sigma2.setZero();
        Eigen::Matrix2d Sigma2_2d;
        if (is_use_3d) {
            Eigen::Matrix3d S3d = R_est * sigmas3d[i] * R_est.transpose();
            Sigma2(0, 0) = S3d(2, 2) * x(0) * x(0) - 2 * S3d(0, 2) * x(0) + S3d(0, 0);
            Sigma2(1, 1) = S3d(2, 2) * x(1) * x(1) - 2 * S3d(1, 2) * x(1) + S3d(1, 1);
            Sigma2(0, 1) = S3d(2, 2) * x(0) * x(1) - S3d(0, 2) * x(1) - S3d(1, 2) * x(0) + S3d(0, 1);
            Sigma2(1, 0) = Sigma2(0, 1);
            Sigma2_2d.setZero();
        } else {
            Sigma2.setZero();
            Sigma2_2d.setIdentity();
        }

        if (is_use_2d)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            double d = Xc(2);
            Sigma2_2d = (d*d*sigmas2d_norm[i] + 1e-6)*Eigen::Matrix2d::Identity();
        }

        Sigma2 = Sigma2 + Sigma2_2d;

        Eigen::Matrix2d Sigma2Inv = Sigma2.inverse();
        S2s.push_back(Sigma2Inv);
        Eigen::Matrix<double,2,3> Ti;
        Ti << -1,0, x(0),
                0, -1, x(1);
        Tps.push_back(Ti);
        T = T + Ti.transpose() * Sigma2Inv*Ti;
        const Eigen::Vector3d& Xp = XX[i];
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0, 0) = -Xp.transpose();
        Ai.block<1,3>(0, 6) = x(0)*Xp.transpose();
        Ai.block<1,3>(1, 3) = -Xp.transpose();
        Ai.block<1,3>(1, 6) = x(1)*Xp.transpose();
        Api.push_back(Ai);
        A = A + Ti.transpose()*Sigma2Inv*Ai;
    }
    std::vector<Eigen::Matrix<double, 2, 9>> Ali;
    Eigen::Matrix<double, 9, 9> A2;
    A2.setZero();

    Eigen::Matrix<double, 3,9> T_express = T.colPivHouseholderQr().solve(-A);
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Api[i] + Tps[i]*T_express;
        A2 = A2 + Ai.transpose() * S2s[i] * Ai;
    }
//    for (int i = 0; i < nl; i++)
//    {
//        Eigen::Matrix<double, 2, 9> Ai = Ali[i]+Tli[i]*T_express;
//        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
//    }

    Matrix<double, 9, 10> QR;
    QR << 1,0,0,0, 1,0,0,-1, 0, -1,
            0,0,0,-2, 0,2,0,0, 0, 0,
            0,0,2,0, 0,0,2,0, 0, 0,
            0,0,0,2, 0,2,0,0, 0, 0,
            1,0,0,0, -1,0,0,1, 0, -1,
            0,-2,0,0, 0,0,0,0, 2, 0,
            0,0,-2,0, 0,0,2,0, 0, 0,
            0,2,0,0, 0,0,0,0, 2, 0,
            1,0,0,0, -1,0,0,-1, 0, 1;
    Matrix<double, 10, 10> A3 = QR.transpose() * A2 * QR;
    VectorXd data(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            data(10*i+j) = A3(j, i);
        }
    }

    MatrixXcd a = solver_opt_pnp_hesch2_red(data);
    sols_p->clear();
    for (int i = 0; i < a.cols(); i++)
    {
        if (fabs(a.col(i)(0).imag()) == 0)
        {
            Vector3d s = a.col(i).real();
            Matrix3d Rc = (1-s.transpose()*s)*Matrix3d::Identity()+2*cpmat(s)+2*s*s.transpose();

            Eigen::Vector3d sp = -s;
            for (int it = 0; it < 1; it++)
            {
                double g[3];
//                void gradient(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* g);
//                void hessian(double a, double b, double c, const Eigen::Matrix<double,9,9>& A2, double* H);
                double H[9];
                gradient(sp(0), sp(1), sp(2), A2, g);
                hessian(sp(0), sp(1), sp(2), A2, H);
                Map<Eigen::Matrix<double, 3, 1> > g_eig(g);
                Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > H_eig(H);
//                std::cout << "hessian " << H_eig << std::endl;
//                std::cout << "gradient " << g_eig << std::endl;
                sp = sp - H_eig.inverse() * g_eig;
            }
//            std::cout << " refined was " << s << " is " << -sp << std::endl;
            Eigen::Vector3d sfin = -sp;
            Rc = 1.0/(1+sfin.transpose()*sfin)*((1-sfin.transpose()*sfin)*Matrix3d::Identity()+2*cpmat(sfin)+2*sfin*sfin.transpose());

//            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            VectorXd rvec(9);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            Vector3d tc = -T.colPivHouseholderQr().solve( A * rvec);
            Matrix4d Tc;
            Tc.setIdentity();
            Tc.block<3,3>(0,0) = Rc;
            Tc.block<3,1>(0,3) = tc;
            sols_p->push_back(Tc);
        }
    }
}


void SolvePnPFullTrueK(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                  const std::vector<Eigen::Vector3d>& Xs, const std::vector<Eigen::Vector3d>& Xe,
                  const std::vector<Eigen::Vector3d>& l, const std::vector<Eigen::Matrix3d>& Sigmas,
                  const mat6dvector& SigmasLines, const mat2dvector& Sigmas2D,
                  const std::vector<Eigen::Matrix3d>& Sigmas2DLines,
                  const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
                  int mode)
{
    int np = XX.size();
    int nl = Xs.size();
    std::vector<double> depths_est;
    std::vector<Eigen::Vector3d> Xsc, Xec;
    std::vector<Eigen::Matrix<double, 2, 3>> Tps;
    std::vector<Eigen::Matrix<double, 3, 3>> Tls;
    Eigen::Matrix3d T;
    T.setZero();
    Eigen::Matrix<double, 3, 9> A;
    A.setZero();
    bool is_est_available = false;
    if (t_est.norm()>0)
    {
        for (int i = 0; i < np; i++)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
        }
        for (int i = 0; i < nl; i++)
        {
            Xsc.push_back(R_est*Xs[i] + t_est);
            Xec.push_back(R_est*Xe[i] + t_est);
        }
        is_est_available = true;
    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;

    double fu = K(0,0);
    double fv = K(1,1);
    double cu = K(0,2);
    double cv = K(1,2);
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix2d Sigma2;
        const Eigen::Matrix3d& S3d = R_est*Sigmas[i]*R_est.transpose();
        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
        const Eigen::Vector2d& x = xh.segment<2>(0);
        Eigen::Vector2d dx = x - Eigen::Vector2d(cu, cv);
        Sigma2(0,0) = S3d(2,2)*dx(0)*dx(0)- 2*S3d(0,2)*dx(0)+S3d(0,0);
        Sigma2(1,1) = S3d(2,2)*dx(1)*dx(1)- 2*S3d(1,2)*dx(1)+S3d(1,1);
        Sigma2(0,1) = S3d(2,2)*dx(0)*dx(1)- S3d(0,2)*dx(1)-S3d(1,2)*dx(0)+S3d(0,1);
        Sigma2(1,0) = Sigma2(0,1);
        if (is_est_available && mode == 0)
        {
            Sigma2 = Sigma2 + depths_est[i]*depths_est[i]*Sigmas2D[i];
        }
        if (is_est_available  && mode == 2)
        {
            Sigma2 = depths_est[i]*depths_est[i]*Sigmas2D[i];
        }
        if (mode == 3)
        {
            Sigma2.setIdentity();
        }
        Eigen::Matrix2d Sigma2Inv = Sigma2.inverse();
        S2s.push_back(Sigma2Inv);
        Eigen::Matrix<double,2,3> Ti;
        Ti << -fu,0, x(0) - cu,
                0, -fv, x(1) -cv;
        Tps.push_back(Ti);
        T = T + Ti.transpose() * Sigma2Inv*Ti;
        const Eigen::Vector3d& Xp = XX[i];
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.setZero();
        Ai.block<1,3>(0, 0) = -fu*Xp.transpose();
        Ai.block<1,3>(0, 6) = (x(0)-cu)*Xp.transpose();
        Ai.block<1,3>(1, 3) = -fv*Xp.transpose();
        Ai.block<1,3>(1, 6) = (x(1)-cv)*Xp.transpose();
        Api.push_back(Ai);
        A = A + Ti.transpose()*Sigma2Inv*Ai;
    }
    std::vector<Eigen::Matrix<double, 2, 9>> Ali;
    std::vector<Eigen::Matrix<double, 2, 3>> Tli;
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix2d Sigma2;
        Eigen::Vector3d lvec = K.transpose() * l[i]; // l'X = 0 => l' K X_n = 0 => (K'l)
        lvec /= lvec.segment<2>(0).norm();
        const Eigen::Matrix<double,6,6>& Sl = SigmasLines[i];
        Sigma2(0,0) = lvec.transpose() * R_est * Sl.block<3,3>(0,0) * R_est.transpose() * lvec;
        Sigma2(0,1) = lvec.transpose() * R_est * Sl.block<3,3>(0,3) * R_est.transpose() * lvec;
        Sigma2(1,1) = lvec.transpose() * R_est * Sl.block<3,3>(3,3) * R_est.transpose() * lvec;
        if (is_est_available)
        {
            Sigma2(0,0) = Sigma2(0,0) + Xsc[i].transpose() * Sigmas2DLines[i] * Xsc[i];
            Sigma2(0,1) = Sigma2(0,1) + Xsc[i].transpose() * Sigmas2DLines[i] * Xec[i];
            Sigma2(1,1) = Sigma2(1,1) + Xec[i].transpose() * Sigmas2DLines[i] * Xec[i];
        }
        Sigma2(1,0) = Sigma2(0,1);
        Eigen::Matrix2d Sigma2Inv = Sigma2.inverse();
        S2ls.push_back(Sigma2Inv);
        Eigen::Matrix<double, 2, 3> Ti;
        Ti.row(0) = lvec.transpose();
        Ti.row(1) = lvec.transpose();
        Tli.push_back(Ti);
        const Eigen::Vector3d& Xsi = Xs[i];
        Eigen::Matrix<double, 3, 9> Xsmat;
        Xsmat.setZero();
        Xsmat.block<1,3>(0,0) = Xsi.transpose();
        Xsmat.block<1,3>(1,3) = Xsi.transpose();
        Xsmat.block<1,3>(2,6) = Xsi.transpose();
        Eigen::Matrix<double, 2, 9> Ai;
        Ai.row(0) = lvec.transpose() * Xsmat;
        const Eigen::Vector3d& Xei = Xe[i];
        Xsmat.block<1,3>(0,0) = Xei.transpose();
        Xsmat.block<1,3>(1,3) = Xei.transpose();
        Xsmat.block<1,3>(2,6) = Xei.transpose();
        Ai.row(1) = lvec.transpose() * Xsmat;
        Ali.push_back(Ai);
        Eigen::Matrix<double, 3, 3> Tl = Ti.transpose()*Sigma2Inv*Ti;
        T = T + Tl;
        Tls.push_back(Tl);
        A = A + Ti.transpose() * Sigma2Inv * Ai;
    }
    Eigen::Matrix<double, 9, 9> A2;
    A2.setZero();

    Eigen::Matrix<double, 3,9> T_express = T.colPivHouseholderQr().solve(-A);
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Api[i] + Tps[i]*T_express;
        A2 = A2 + Ai.transpose() * S2s[i] * Ai;
    }
    for (int i = 0; i < nl; i++)
    {
        Eigen::Matrix<double, 2, 9> Ai = Ali[i]+Tli[i]*T_express;
        A2 = A2 + Ai.transpose() * S2ls[i] * Ai;
    }

    Matrix<double, 9, 10> QR;
    QR << 1,0,0,0, 1,0,0,-1, 0, -1,
            0,0,0,-2, 0,2,0,0, 0, 0,
            0,0,2,0, 0,0,2,0, 0, 0,
            0,0,0,2, 0,2,0,0, 0, 0,
            1,0,0,0, -1,0,0,1, 0, -1,
            0,-2,0,0, 0,0,0,0, 2, 0,
            0,0,-2,0, 0,0,2,0, 0, 0,
            0,2,0,0, 0,0,0,0, 2, 0,
            1,0,0,0, -1,0,0,-1, 0, 1;
    Matrix<double, 10, 10> A3 = QR.transpose() * A2 * QR;
    VectorXd data(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            data(10*i+j) = A3(j, i);
        }
    }

    MatrixXcd a = solver_opt_pnp_hesch2_red(data);
    sols_p->clear();
    for (int i = 0; i < a.cols(); i++)
    {
        if (fabs(a.col(i)(0).imag()) == 0)
        {
            Vector3d s = a.col(i).real();
            Matrix3d Rc = (1-s.transpose()*s)*Matrix3d::Identity()+2*cpmat(s)+2*s*s.transpose();
            Rc = Rc / pow(Rc.determinant(), 1.0/3.0);
            VectorXd rvec(9);
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    rvec(3*ii+jj) = Rc(ii, jj);
                }
            }
            Vector3d tc = -T.colPivHouseholderQr().solve( A * rvec);
            Matrix4d Tc;
            Tc.setIdentity();
            Tc.block<3,3>(0,0) = Rc;
            Tc.block<3,1>(0,3) = tc;
            sols_p->push_back(Tc);
        }
    }
}


int FindBestSolutionReproj(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                           const Eigen::Matrix3d& K,
                           const posevector& poses, double* min_err_p)
{
    double min_err = 1e100;
    int sol_ind = -1;
    for (int i = 0; i < poses.size(); i++)
    {
        auto T = poses[i];
        double err = 0;
        double cnt = 0.0;
        bool is_behind = false;
        for (int j = 0; j < XX.size(); j++)
        {
            Eigen::Vector3d Xp = XX[j];
            Eigen::Vector3d xc = K*(T.block<3,3>(0,0) * Xp + T.block<3,1>(0,3));
            if (xc(2) < 0)
            {
                is_behind = true;
            }
            xc = xc / xc(2);
            err += (xc.segment<2>(0) - xx[j]).norm();
            cnt += 1;
        }
        err = err/cnt;
        if (err < min_err && !is_behind)
        {
            min_err = err;
            sol_ind = i;
        }
    }
    *min_err_p = min_err;
    return sol_ind;
}

void RobustDLSU(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                const std::vector<float>& sigmas3d,
                const std::vector<float>& sigmas2d_norm,
                const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
                bool is_use_3d, bool is_use_2d)
{
    DLSU(XX, xx, sigmas3d, sigmas2d_norm, R_est, t_est, sols_p, is_use_3d, is_use_2d);
    double min_err;
    int sol_ind = FindBestSolutionReproj(XX, xx, K, *sols_p, &min_err);
    int cnt = 0;
    while (sol_ind < 0 && cnt < 5)
    {
        cv::Mat rvec(3, 1, CV_64FC1);
        cv::theRNG().fill(rvec, cv::RNG::NORMAL, 0, 6.28);
        cv::Mat Rr;
        cv::Rodrigues(rvec, Rr);
        Eigen::Matrix3d Rr_eig;
        cv::cv2eigen(Rr, Rr_eig);
//            std::cout << " random rot " << Rr_eig << std::endl;
        std::vector<Eigen::Vector3d> XXr(XX.size());

        for (int k = 0; k < XX.size(); k++) {
            XXr[k] = Rr_eig * XX[k];
        }

        DLSU(XXr, xx, sigmas3d, sigmas2d_norm, R_est * Rr_eig.transpose(), t_est, sols_p, is_use_3d, is_use_2d);

        for (int k = 0; k < sols_p->size(); k++) {
            (*sols_p)[k].block<3, 3>(0, 0) = (*sols_p)[k].block<3, 3>(0, 0) * Rr_eig;
        }

        sol_ind = FindBestSolutionReproj(XX, xx, K, *sols_p, &min_err);

        cnt++;
    }
}

void RobustDLSU_accurate(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                         const std::vector<Eigen::Matrix3d>& sigmas3d,
                         const std::vector<float>& sigmas2d_norm,
                         const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p,
                         bool is_use_3d, bool is_use_2d)
{
    DLSU_accurate(XX, xx, sigmas3d, sigmas2d_norm, R_est, t_est, sols_p, is_use_3d, is_use_2d);
    double min_err;
    int sol_ind = FindBestSolutionReproj(XX, xx, K, *sols_p, &min_err);
    int cnt = 0;
    while (sol_ind < 0 && cnt < 5)
    {
        cv::Mat rvec(3, 1, CV_64FC1);
        cv::theRNG().fill(rvec, cv::RNG::NORMAL, 0, 6.28);
        cv::Mat Rr;
        cv::Rodrigues(rvec, Rr);
        Eigen::Matrix3d Rr_eig;
        cv::cv2eigen(Rr, Rr_eig);
//            std::cout << " random rot " << Rr_eig << std::endl;
        std::vector<Eigen::Vector3d> XXr(XX.size());
        std::vector<Eigen::Matrix3d> sigmas3dr(XX.size());

        for (int k = 0; k < XX.size(); k++) {
            XXr[k] = Rr_eig * XX[k];
            sigmas3dr[k] = Rr_eig * sigmas3d[k] * Rr_eig.transpose();
        }

        DLSU_accurate(XXr, xx, sigmas3dr, sigmas2d_norm, R_est * Rr_eig.transpose(), t_est, sols_p, is_use_3d, is_use_2d);

        for (int k = 0; k < sols_p->size(); k++) {
            (*sols_p)[k].block<3, 3>(0, 0) = (*sols_p)[k].block<3, 3>(0, 0) * Rr_eig;
        }

        sol_ind = FindBestSolutionReproj(XX, xx, K, *sols_p, &min_err);

        cnt++;
    }
}