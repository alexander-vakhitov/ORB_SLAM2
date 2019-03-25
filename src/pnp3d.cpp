//
// Created by alexander on 17.08.18.
//
#include "pnp3d.h"
#include <Eigen/Dense>
#include <iostream>
#include "pnp3d_newton.h"
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

void DLSU(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                  const std::vector<float>& sigmas3d,
                  const std::vector<float>& sigmas2d_norm,
                  const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p)
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
    if (t_est.norm()>0)
    {
        for (int i = 0; i < np; i++)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
        }
        is_est_available = true;
    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix2d Sigma2;
        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
        float s3d = sigmas3d[i];
        Sigma2(0,0) = s3d*(x(0)*x(0)+1);
        Sigma2(1,1) = s3d*(x(1)*x(1)+1);
        Sigma2(0,1) = s3d*x(0)*x(1);
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

void DLSULines(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
               const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
               const std::vector<Eigen::Vector3d>& l2ds,
               const std::vector<float>& sigmas3d,
               const std::vector<float>& sigmas2d_norm,
               const mat6dvector& sigmas3d_lines,
               const std::vector<Eigen::Matrix3d>& sigmas2d_lines,
               const std::vector<float>& sigmasDetLines,
               const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p)
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
    if (t_est.norm()>0)
    {
        for (int i = 0; i < np; i++)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
        }
        is_est_available = true;
    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api, Ali;
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix2d Sigma2;
        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
        float s3d = sigmas3d[i];
        Sigma2(0,0) = s3d*(x(0)*x(0)+1);
        Sigma2(1,1) = s3d*(x(1)*x(1)+1);
        Sigma2(0,1) = s3d*x(0)*x(1);
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

        Sigma2_2d.setIdentity();
        Sigma2_2d(0,0) = Xsc(2)*Xsc(2);
        Sigma2_2d(1,1) = Xec(2)*Xec(2);
        Sigma2_2d *= sigmasDetLines[i]/K(0,0)/K(0,0);
        Sigma2 = Sigma2_2d+Sigma2;
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

void DLSULines_accurate(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx,
                        const std::vector<Eigen::Vector3d>& XXs, const std::vector<Eigen::Vector3d>& XXe,
                        const std::vector<Eigen::Vector3d>& l2ds,
                        const std::vector<Eigen::Matrix3d>& sigmas3d,
                        const std::vector<float>& sigmas2d_norm,
                        const mat6dvector& sigmas3d_lines,
                        const std::vector<Eigen::Matrix3d>& sigmas2d_lines,
                        const std::vector<float>& sigmasDetLines,
                        const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est,
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
    if (t_est.norm()>0)
    {
        for (int i = 0; i < np; i++)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
        }
        is_est_available = true;
    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix2d Sigma2;
        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
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
//        Sigma2_2d(0,0) = Xsc.transpose()*s2d*Xsc;
//        Sigma2_2d(1,1) = Xec.transpose()*s2d*Xec;
//        Sigma2_2d(0,1) = Xsc.transpose()*s2d*Xec;
//        Sigma2_2d(1,0) = Xec.transpose()*s2d*Xsc;
        Sigma2_2d.setIdentity();
        Sigma2_2d(0,0) = Xsc(2)*Xsc(2);
        Sigma2_2d(1,1) = Xec(2)*Xec(2);
        Sigma2_2d *= sigmasDetLines[i]/K(0,0)/K(0,0);
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

    std::vector<float> s2ls;
    std::vector<Eigen::Matrix<double,1,9>> alis;
    for (int i = 0; i < nl; i++)
    {
        s2ls.push_back(1.0/(sigmasDetLines[i] + l2ds[i].transpose()*(R_est*sigmas3d_lines[i].block<3,3>(3,3)*R_est.transpose())*l2ds[i]));
        Eigen::Matrix<double,3,9> Xi;
        Eigen::Matrix<double, 3, 9> Xsi;
        Xsi.setZero();
        const Eigen::Vector3d& Xs = XXe[i]-XXs[i];
        Xsi.block<1,3>(0,0) = Xs.transpose();
        Xsi.block<1,3>(1,3) = Xs.transpose();
        Xsi.block<1,3>(2,6) = Xs.transpose();
        alis.push_back(l2ds[i].transpose() * Xsi);
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
        A2 = A2 + alis[i].transpose() * s2ls[i] * alis[i];
    }


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
          const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, posevector* sols_p)
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
    if (t_est.norm()>0)
    {
        for (int i = 0; i < np; i++)
        {
            Eigen::Vector3d Xc = R_est*XX[i] + t_est;
            depths_est.push_back(Xc(2));
//            std::cout << " de = " << depths_est[i] << std::endl;
        }
        is_est_available = true;
    }
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> S2s, S2ls;
    std::vector<Eigen::Matrix<double, 2, 9>> Api;
    for (int i = 0; i < np; i++)
    {
        Eigen::Matrix2d Sigma2;
        Eigen::Vector3d xh (xx[i](0), xx[i](1), 1.0);
        const Eigen::Vector2d& x = (K.inverse() * xh).segment<2>(0);
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
