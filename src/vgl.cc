//
// Created by alexander on 12.12.17.
//
#include "vgl.h"
#include <iostream>
#include <opencv/cv.hpp>
#include <opencv/cxeigen.hpp>


bool vgl::MultiTriangulateLine(const posevector& Ts,
                          const std::vector<Eigen::Vector3d>& lines,
                          Eigen::Vector3d* X0_p, Eigen::Vector3d* line_dir_p)
{
    if (lines.size() < 3 || lines.size() != Ts.size())
    {
        return false;
    }
    std::vector<Eigen::Vector3d> normals;
    for (int i = 0; i < lines.size(); i++)
    {
        Eigen::Vector3d leq = lines[i];
        leq = leq/leq.norm();
        normals.push_back(Ts[i].block<3,3>(0,0)*leq);
    }

    for (int i = 1; i < normals.size(); i++)
    {
        if (fabs(normals[0].dot(normals[i]))/normals[0].norm() / normals[i].norm()>0.99)
        {
            return false;
        }
    }

    Eigen::MatrixXd M2(lines.size(), 3);
    for (int i = 0; i < lines.size(); i++)
    {
        M2.row(i) = normals[i].transpose();
    }
    Eigen::Matrix3d Vm = M2.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).matrixV();
    Eigen::Vector3d line_dir = Vm.col(2);

    Eigen::MatrixXd M1(lines.size(), 3);
    Eigen::VectorXd b1(lines.size());
    for (int i = 0; i < lines.size(); i++)
    {
        Eigen::Vector3d leq = lines[i];
        M1.row(i) = normals[i].transpose();
        b1(i) = normals[i].dot(Ts[i].block<3,1>(0,3));
    }
//    M1.row(lines.size()) = line_dir.transpose();
//    b1(lines.size()) = 0;
    Eigen::Vector3d X0 = M1.colPivHouseholderQr().solve(b1);

    X0 = X0 - (X0.dot(line_dir))*line_dir;
    *line_dir_p = line_dir;
    *X0_p = X0;
    return true;
}

Eigen::Vector3d GetSegmentLine(const vec2d& seg)
{
    Eigen::Vector3d xs(seg[0](0), seg[0](1), 1.0);
    Eigen::Vector3d xe(seg[1](0), seg[1](1), 1.0);
    return xs.cross(xe);
}

bool CheckSegmentEndpoints(const PMatrix& P1, const vec2d& seg,
        const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir)
{
    Eigen::Vector3d X,Y;
    vgl::ReprojectEndpointTo3DProjMat(P1, seg[0], X0, line_dir, &X);
    //test
    Eigen::Vector3d Xc = P1.block<3,3>(0,0)*X + P1.block<3,1>(0,3);
//    std::cout << "est proj " << Xc/Xc(2)  << " true proj " << seg.first << " pt " << Xc << std::endl;
    vgl::ReprojectEndpointTo3DProjMat(P1, seg[1], X0, line_dir, &Y);
    Eigen::Vector3d Yc = P1.block<3,3>(0,0)*Y + P1.block<3,1>(0,3);
//    std::cout << "est proj " << Yc/Yc(2)  << " true proj " << seg.second << " pt " << Yc << std::endl;
    return (Xc(2)>0 && Yc(2)>0);
}

bool vgl::TriangulateLineProjective(const PMatrix& P1,
                          const PMatrix& P2,
                          const vec2d& segment1,
                          const vec2d& segment2,
                          Eigen::Vector3d* X0, Eigen::Vector3d* line_dir)
{
    Eigen::Vector3d l1 = GetSegmentLine(segment1);
    Eigen::Vector3d l2 = GetSegmentLine(segment2);
    Eigen::Matrix<double, 2, 3> A;
    Eigen::Vector2d b;
    A.block<1,3>(0,0) = l1.transpose() * (P1.block<3,3>(0,0));
    A.block<1,3>(1,0) = l2.transpose() * (P2.block<3,3>(0,0));
    Eigen::Matrix3d AtA = A.transpose() * A;
    b(0) = - l1.dot(P1.block<3,1>(0,3));
    b(1) = - l2.dot(P2.block<3,1>(0,3));
    Eigen::Vector3d Atb = A.transpose() * b;
    auto svd = AtA.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
//    std::cout << svd.matrixU()*Eigen::DiagonalMatrix<double,3>(svd.singularValues()[0], svd.singularValues()[1], svd.singularValues()[2])*svd.matrixV().transpose() << std::endl;
//    std::cout << AtA << std::endl;
    Eigen::DiagonalMatrix<double,3> DiagMat(1.0/svd.singularValues()[0], 1.0/svd.singularValues()[1], 0.0);
    Eigen::Vector3d X_start = svd.matrixV() * DiagMat * svd.matrixU().transpose() * Atb;
    *line_dir = svd.matrixV().col(2);
    double xsp = (X_start.dot(*line_dir));
    *X0 = X_start - xsp * (*line_dir);
//    std::cout << "est X0 " << *X0 << std::endl;
//    std::cout << "est linedir " << *line_dir << std::endl;
    return CheckSegmentEndpoints(P1, segment1, *X0, *line_dir) && CheckSegmentEndpoints(P2, segment2, *X0, *line_dir);
}

bool vgl::TriangulateLine(const Eigen::Matrix4d& T1,
                     const Eigen::Matrix4d& T2, const Eigen::Vector3d& line2d_n_1,
                     const Eigen::Vector3d& line2d_n_2, Eigen::Vector3d* X0, Eigen::Vector3d* line_dir)
{
    Eigen::Vector3d normal_1 = T1.block<3,3>(0,0) * line2d_n_1;
    Eigen::Vector3d normal_2 = T2.block<3,3>(0,0) * line2d_n_2;
    Eigen::Matrix<double, 2, 3> M2;
    M2.row(0) = normal_1.transpose();
    M2.row(1) = normal_2.transpose();
    auto svd = M2.jacobiSvd(Eigen::DecompositionOptions::ComputeFullV|Eigen::DecompositionOptions::ComputeFullU);
    Eigen::Matrix3d Vmat = svd.matrixV();
    *line_dir = Vmat.col(2);

    if (fabs(normal_1.dot(normal_2)) / normal_1.norm() / normal_2.norm() > 0.99)
    {
        return false;
    }

    auto svs = svd.singularValues();
    Eigen::Matrix3d S;
    S.setZero();
    S(0,0) = 1.0/svs(0);
    S(1,1) = 1.0/svs(1);
//    std::cout << svd.matrixU().cols() << " " << svd.matrixU().rows() << std::endl;
    Eigen::Matrix2d Umat = svd.matrixU();
    Eigen::Vector3d b;
    b(0) = normal_1.transpose() * T1.block<3,1>(0, 3);
    b(1) = normal_2.transpose() * T2.block<3,1>(0, 3);
    b(2) = 0.0;
    Eigen::Vector3d b0 = b;
    b.segment<2>(0) = Umat.transpose() * b.segment<2>(0);
//    Eigen::Vector3d b_ext;
    *X0 = Vmat * S * b;
    *X0 = *X0 - (*X0).dot(*line_dir) * (*line_dir);
//    std::cout << " m2 res " << M2 * (*line_dir) << std::endl;
//    std::cout << " m2 res2 " << M2 * (*X0) - b0.segment<2>(0) << std::endl;
//    Eigen::Matrix3d M;
//    Eigen::Vector3d b;
//    M.row(0) = normal_1.transpose();
//    M.row(1) = normal_2.transpose();
//    b(0) = normal_1.transpose() * T1.block<3,1>(0, 3);
//    b(1) = normal_2.transpose() * T2.block<3,1>(0, 3);
//    b(2) = 0;
//    M.row(2) = line_dir->transpose();
//    auto qr = M.colPivHouseholderQr();
//    if (qr.rank() < 3)
//    {
//        return false;
//    }
//    *X0 = qr.solve(b);

//    double res = line2d_n_2.transpose() * T2.block<3,3>(0,0).transpose() * (*X0 + *line_dir - T2.block<3,1>(0,3) );
//    std::cout << res << std::endl;
//    std::cout << line2d_n_1.transpose() * T1.block<3,3>(0,0).transpose() * (*X0 + *line_dir - T1.block<3,1>(0,3 ) << std::endl;
//
//    std::cout << line2d_n_2.transpose() * T2.block<3,3>(0,0).transpose() * (*X0 - T2.block<3,1>(0,3) ) << std::endl;
//    std::cout << line2d_n_2.transpose() * T2.block<3,3>(0,0).transpose() * (*X0 + *line_dir - T2.block<3,1>(0,3) ) << std::endl;

    return true;
//    Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullV);
//    *X0 = svd.matrixV().col(2);


//    Eigen::Vector3d test0 = M * (*X0);

//    std::cout << test0.norm() << std::endl;

}

int GetAxisCode(const Eigen::Vector3d& line_dir)
{
    Eigen::Matrix3d rem = Eigen::Matrix3d::Identity() - line_dir * line_dir.transpose();
    int ax_code = -1;
    double abs_proj = 0;
    Eigen::Vector3d ax;
    do
    {
        ax_code++;
        ax = Eigen::Vector3d::Zero();
        ax(2-ax_code) = 1.0;
        abs_proj = (rem * ax).norm();
    } while (abs_proj < 1e-8);
    return ax_code;
}

void GetAxes(int ax_code, const Eigen::Vector3d& line_dir, Eigen::Vector3d* x_axis, Eigen::Vector3d* y_axis)
{
    Eigen::Vector3d ax;
    ax.setZero();
    ax(ax_code) = 1.0;
    *x_axis = ax - line_dir * (line_dir.transpose() * ax);
    *x_axis = *x_axis / (x_axis->norm());
    *y_axis = line_dir.cross(*x_axis);
}

void vgl::Line3DFromPluecker(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, int* ax_code_p, double* coords_p)
{
    double b = atan2(line_dir(1), line_dir(0));
    double cos_a_mod = line_dir.segment<2>(0).norm();
    double a = asin(line_dir(2)); // cos(a) always positive

    int ax_code = GetAxisCode(line_dir);
    Eigen::Vector3d x_axis, y_axis;
    GetAxes(ax_code, line_dir, &x_axis, &y_axis);
    double x = x_axis.dot(X0);
    double y = y_axis.dot(X0);

    coords_p[0] = a;
    coords_p[1] = b;
    coords_p[2] = x;
    coords_p[3] = y;
    *ax_code_p = ax_code;
}

void vgl::PlueckerFromLine3D(const double coords[4], int ax_code, Eigen::Vector3d* X0, Eigen::Vector3d* line_dir)
{
    double a = coords[0];
    double b = coords[1];
    *line_dir << cos(a)*cos(b), cos(a)*sin(b), sin(a);
    double x = coords[2];
    double y = coords[3];
    Eigen::Vector3d x_axis, y_axis;
    GetAxes(ax_code, *line_dir, &x_axis, &y_axis);
    *X0 = x*x_axis + y*y_axis;
}

void vgl::RLine3DFromPluecker(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, double* coords)
{
    cv::Mat R(3,3,CV_64FC1);
    cv::Mat r1(3,1,CV_64FC1);
    cv::eigen2cv(line_dir, r1);
    cv::Mat r2(3,1,CV_64FC1);
    double alpha = X0.norm();
    Eigen::Vector3d X0n = X0/alpha;
    cv::eigen2cv(X0n, r2);
    cv::Mat r3 = r1.cross(r2);

    r1.copyTo(R.col(0));
    r2.copyTo(R.col(1));
    r3.copyTo(R.col(2));
    cv::Mat rvec;
//    std::cout << R << std::endl;
    cv::Rodrigues(R, rvec);
    cv::Mat Rtest;
    cv::Rodrigues(rvec, Rtest);
//    std::cout << cv::norm(R-Rtest) << std::endl;
    for (int i = 0; i < 3; i++)
    {
        coords[i] = rvec.at<double>(i,0);
    }
    coords[3] = alpha;
}

void vgl::PlueckerFromRLine3D(const double coords[4], Eigen::Vector3d* X0, Eigen::Vector3d* line_dir, Eigen::Matrix4d* J_X0, Eigen::Matrix4d* J_line_dir)
{
    cv::Mat rvec(3, 1, CV_64FC1);
    for (int i = 0; i < 3; i++)
    {
        rvec.at<double>(i, 0) = coords[i];
    }
    cv::Mat R, JR;
    cv::Rodrigues(rvec, R, JR);
    cv::Mat r1 = R.col(0);
    cv::cv2eigen(r1, *line_dir);
    cv::Mat r2 = R.col(1);
    cv::cv2eigen(r2, *X0);
    *X0 = *X0 * coords[3];
    if (J_X0)
    {
        //assume column-major order of R
//        cv::Mat J_X0_theta_cv = coords[3]*JR(cv::Rect(3,0,3,3)).t();
//        Eigen::Matrix3d J_X0_theta;
//        cv::cv2eigen(J_X0_theta_cv, J_X0_theta);
//        J_X0->block<3,3>(0,0) = J_X0_theta;
//        Eigen::Vector3d r2_eig;
//        cv::cv2eigen(r2, r2_eig);
//        J_X0->block<3,1>(0,3) = r2_eig;
//
//        cv::Mat J_d_theta_cv = JR(cv::Rect(0,0,3,3)).t();
//        Eigen::Matrix3d J_d_theta;
//        cv::cv2eigen(J_d_theta_cv, J_d_theta);
//        J_line_dir->block<3,3>(0,0) = J_d_theta;
//        J_line_dir->block<3,1>(0,3) = Eigen::Vector3d::Zero();
    }
}

bool vgl::IsLineCrossingRect(const cv::Point2f& lu, const cv::Point2f& rd, const Eigen::Vector3d& line_eq,
                             std::vector<Eigen::Vector3d>* pps, bool segment_priority)
{
    std::vector<Eigen::Vector3d> corners;
    int ci = 0;
    int cj = 0;
    for (int ti = 0; ti < 4; ti++)
    {
        if (ti % 2 == 0)
        {
            ci += 1;
        } else {
            cj += 1;
        }
        int ri = ci % 2;
        int rj = cj % 2;
        Eigen::Vector3d v(lu.x + (rd.x-lu.x)*(ri+0.0), lu.y + (rd.y-lu.y)*(rj+0.0), 1.0);
        corners.push_back(v );
    }
    if (pps != NULL && !segment_priority)
    {
        pps->clear();
    }

    bool rv = false;
    Eigen::Vector3d center_pt(0.5*(lu.x+rd.x), 0.5*(lu.y+rd.y), 1.0);
    for (int i = 0; i < 4; i++)
    {
        Eigen::Vector3d v1 = corners[i];
        int j = i+1;
        if (j == 4)
        {
            j = 0;
        }
        Eigen::Vector3d v2 = corners[j];
        double p1 = (line_eq.transpose() * v1);
        double p2 = (line_eq.transpose() * v2);
        if (p1 * p2 < 0)
        {
            Eigen::Vector3d side_vec = v1.cross(v2);
            Eigen::Vector3d pp = side_vec.cross(line_eq);
            if (pps != NULL)
            {
                if (segment_priority)
                {
                    int pt_id = -1;
                    for (int k = 0; k < 2; k++)
                    {
                        double p1 = (side_vec.transpose() * (*pps)[k]);
                        double p2 = (side_vec.transpose() * center_pt);
                        if (p1 * p2 < 0)
                        {
                            if (pt_id < 0)
                            {
                                pt_id = k;
                            } else {
                                rv = false;
                                return rv;
                            }
                        }
                    }
                    if (pt_id >= 0)
                    {
                        (*pps)[pt_id] = pp/pp(2);
                    }
                } else {
                    pps->push_back(pp / pp(2));
                }
            }
            rv = true;
        }
    }

    return rv;
}



bool vgl::IsLineProjectedToCamera(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, const Eigen::Matrix3d& K_inv,
                                  const Eigen::Matrix4d& T, const cv::Size& im_size, Eigen::Vector3d* line_eq, std::vector<Eigen::Vector3d>* pps)
{

    ProjectLine(X0, line_dir, T, line_eq);

    Eigen::Vector3d p1 = T.block<3,3>(0,0).transpose() * (X0 - T.block<3,1>(0,3));
    Eigen::Vector3d dir = T.block<3,3>(0,0).transpose() * line_dir;

    Eigen::Vector3d lu = K_inv * Eigen::Vector3d(0,0,1);
    Eigen::Vector3d rd = K_inv * Eigen::Vector3d(im_size.width,im_size.height,1);
    bool rv = IsLineCrossingRect(cv::Point2f(lu(0), lu(1)), cv::Point2f(rd(0), rd(1)), *line_eq, pps);
    if (pps == NULL)
    {
        return rv;
    }
    rv = false;
    for (int i = 0; i < pps->size(); i++)
    {

        double depth;
        double line_param;
        Eigen::Vector3d pp = (*pps)[i];
        ReprojectLinePointTo3D(p1, dir, pp.segment<2>(0), Eigen::Matrix3d::Identity(), &depth, &line_param);
        if (pp(2)*depth > 0)
        {
            rv = true;
        }
    }
    return rv;
}


void vgl::ReprojectLinePointTo3D(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, const Eigen::Vector2d& pp, const Eigen::Matrix3d& K,
                            double* depth, double* line_param)
{
    Eigen::Matrix<double, 3, 2> M;
    M.col(1) = - K*line_dir;
    Eigen::Vector3d pp_h (pp(0), pp(1), 1.0);
    M.col(0) = pp_h;
    Eigen::Matrix<double, 2, 1> sol = M.colPivHouseholderQr().solve(K*X0);
    *depth = sol(0);
    *line_param = sol(1);
}

void vgl::ProjectLine(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, const Eigen::Matrix4d& T, Eigen::Vector3d* line_eq)
{
    Eigen::Vector3d p1 = T.block<3,3>(0,0).transpose() * (X0 - T.block<3,1>(0,3));
    Eigen::Vector3d p2 = T.block<3,3>(0,0).transpose() * (X0+line_dir - T.block<3,1>(0,3));
    Eigen::Vector3d dir = T.block<3,3>(0,0).transpose() * line_dir;
    *line_eq = p1.cross(p2);
    *line_eq = (*line_eq)/line_eq->segment<2>(0).norm();
}

void GetMinMax(const int& a, const int& b, int* min_p, int* max_p)
{
    int min_val = std::min(a,b);
    int max_val = std::max(a,b);
    *max_p = max_val;
    *min_p = min_val;
}


//bool vgl::ReprojBehind(const Eigen::Vector3d& xs3, const Eigen::Vector3d& xe3, const Eigen::Matrix<double, 4, 4>& T_other, const Eigen::Vector3d& X0,
//                            const Eigen::Vector3d& line_dir)
//{
//    Eigen::Vector3d X0c = T_other.block<3, 3>(0, 0).transpose() * (X0 - T_other.block<3, 1>(0, 3));
//    Eigen::Vector3d line_dir_c = T_other.block<3, 3>(0, 0).transpose() * line_dir;
//    double d3, p3;
//    ReprojectLinePointTo3D(X0c, line_dir_c, xs3, &d3, &p3);
//    double d4, p4;
//    ReprojectLinePointTo3D(X0c, line_dir_c, xe3, &d4, &p4);
//}
void GetYSortedEndPoints(const KeyLine& kl1, int* x1_s, int* x1_e, int* y1_s, int* y1_e)
{
    if (kl1.startPointY < kl1.endPointY)
    {
        *x1_s = kl1.startPointX;
        *y1_s = kl1.startPointY;
        *x1_e = kl1.endPointX;
        *y1_e = kl1.endPointY;
    } else {
        *x1_s = kl1.endPointX;
        *y1_s = kl1.endPointY;
        *x1_e = kl1.startPointX;
        *y1_e = kl1.startPointY;
    }
}

double GetXByY(const int& x_s, const int& x_e, const int& y_s, const int& y_e, int y_ref)
{
    return (double(y_ref-y_s) * x_e + double (y_e - y_ref) * x_s)/ double(y_e-y_s);
}

bool RefinePoint(const double& x_left, const double& x_right_init, const int& y, const cv::Mat& im_left,
                 const cv::Mat& im_right, const int& w, const int& L, double* x_right_fin)
{
//    const int w = 5;
//    const int L = 10;
    if (y < w || y+w >= im_left.rows)
    {
        return false;
    }

    int x_left_i = int(floor(x_left+0.5));
    if (x_left_i < w || x_left_i+w >= im_left.cols)
    {
        return false;
    }
    int x_right_i = int(floor(x_right_init + 0.5));

    if (x_right_i-L < w || x_right_i+L+w >= im_left.cols)
    {
        return false;
    }
    cv::Mat IL = im_left.rowRange(y-w,y+w+1).colRange(x_left_i - w, x_left_i + w + 1);
    IL.convertTo(IL,CV_32F);
    IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

    int bestDist = INT_MAX;
    int bestincR = 0;

    std::vector<float> vDists;
    vDists.resize(2*L+1);


    const float iniu = x_right_i + L - w;
    const float endu = x_right_i + L + w + 1;

    bool rv = true;
    if(iniu<0 || endu >= im_right.cols) {
//        std::cout << " case 2 " << std::endl;
        rv = false;
    }

    for(int incR=-L; incR<=+L; incR++)
    {
        cv::Mat IR = im_right.rowRange(y-w,y+w+1).colRange(x_right_i + incR - w,x_right_i + incR + w + 1);
        IR.convertTo(IR,CV_32F);
        IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

        float dist = cv::norm(IL,IR,cv::NORM_L1);
        if(dist<bestDist)
        {
            bestDist =  dist;
            bestincR = incR;
        }

        vDists[L+incR] = dist;
    }

    if(bestincR==-L || bestincR==L) {
//        std::cout << " case 1 " << std::endl;
        rv = false;
    }

    const float dist1 = vDists[L+bestincR-1];
    const float dist2 = vDists[L+bestincR];
    const float dist3 = vDists[L+bestincR+1];

    const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

    if(deltaR<-1 || deltaR>1) {
//        std::cout << " case 3 " << std::endl;
        rv = false;
    }

    if (!rv) {
//        std::cout << " costs:" << std::endl;
//        for (int i = 0; i < vDists.size(); i++)
//        {
//            std::cout << vDists[i] << " ";
//        }
//        std::cout << std::endl;
//        cv::Mat iml_d, imr_d;
//        cv::cvtColor(im_left, iml_d, CV_GRAY2BGR);
//        cv::cvtColor(im_right, imr_d, CV_GRAY2BGR);
//        cv::rectangle(iml_d, cv::Point(x_left - w, y - w), cv::Point(x_left + w + 1, y + w + 1), cv::Scalar(0, 0, 255));
//        cv::rectangle(imr_d, cv::Point(x_right - w - L, y - w), cv::Point(x_right + L + w + 1, y + w + 1),
//                      cv::Scalar(0, 255, 0));
//        cv::imshow("iml", iml_d);
//        cv::imshow("imr", imr_d);
//        cv::waitKey(0);
    }

    double dx = x_left - x_left_i;
    double x_for_i = x_right_i + deltaR + bestincR;
    *x_right_fin = x_for_i + dx;
    return rv;
}


//works only for integer endpoint coordinates
//bool vgl::RefineLineStereo(const KeyLine& kl1, const KeyLine& kl2, const cv::Mat& frame1, const cv::Mat& frame2, KeyLine* kl1_ref_p, KeyLine* kl2_ref_p)
//{
//    int x1_s, y1_s, x1_e, y1_e;
//    GetYSortedEndPoints(kl1, &x1_s, &x1_e, &y1_s, &y1_e);
//    int x2_s, y2_s, x2_e, y2_e;
//    GetYSortedEndPoints(kl2, &x2_s, &x2_e, &y2_s, &y2_e);
//
//    int y_l = std::max(y1_s, y2_s);
//    int y_h = std::min(y1_e, y2_e);
//
//    if (y_l > y_h)
//    {
//        return false;
//    }
//
//    if (y_h-y_l < 10)
//    {
//        return false;
//    }
//
//    const int w = 5;
//    const int w_half = (w-1)/2;
//    const int L = 3;
//
//    int y_ref_start = y_l + w_half;
//    int y_ref_end = y_h-w_half;
//
//    double x1_ref_start = GetXByY(x1_s, x1_e, y1_s, y1_e, y_ref_start);
//    double x1_ref_end = GetXByY(x1_s, x1_e, y1_s, y1_e, y_ref_end);
//    double x2_ref_start = GetXByY(x2_s, x2_e, y2_s, y2_e, y_ref_start);
//    double x2_ref_end = GetXByY(x2_s, x2_e, y2_s, y2_e, y_ref_end);
//
//
//
//    double x2_start, x2_end;
//    bool rv_start = RefinePoint(x1_ref_start, x2_ref_start, y_ref_start, frame1, frame2, w, L, &x2_start);
//    bool rv_end = RefinePoint(x1_ref_end, x2_ref_end, y_ref_end, frame1, frame2, w, L, &x2_end);
//    KeyLine kl_f = kl2;
//    kl_f.startPointY = y_ref_start;
//    kl_f.endPointY = y_ref_end;
////    kl_f.startPointX = x2_ref_start;//x2_start;
////    kl_f.endPointX = x2_ref_end;//x2_end;
//    kl_f.startPointX = x2_start;
//    kl_f.endPointX = x2_end;
//
//    KeyLine kl_f_1 = kl1;
//    kl_f_1.startPointY = y_ref_start;
//    kl_f_1.endPointY = y_ref_end;
//    kl_f_1.startPointX = x1_ref_start;
//    kl_f_1.endPointX = x1_ref_end;
//    *kl2_ref_p = kl_f;
//    *kl1_ref_p = kl_f_1;
//
////    return true;
//
//    return rv_start && rv_end;
//}


double vgl::LineEptReprojError(const Eigen::Vector3d& l, const Eigen::Matrix<double, 4, 4>& T, const Eigen::Vector3d& X1, const Eigen::Vector3d& X2, double f)
{
    Eigen::Vector3d p1c = T.block<3,3>(0,0).transpose() * (X1 - T.block<3,1>(0,3));
    Eigen::Vector3d p2c = T.block<3,3>(0,0).transpose() * (X2 - T.block<3,1>(0,3));
    return f * (fabs(1.0/p1c(2) * l.dot(p1c)) + fabs(1.0/p2c(2) * l.dot(p2c)));
}



double vgl::LineReprojErrorL1(const Eigen::Vector2d& xs, const Eigen::Vector2d& xe, const Eigen::Matrix<double, 4, 4>& T_other, const Eigen::Vector3d& X0,
                       const Eigen::Vector3d& line_dir, const Eigen::Matrix3d& K)
{
    Eigen::Vector3d Xc1 = K*MapPoint(T_other, X0);
    Eigen::Vector3d Xc2 = K*MapPoint(T_other, X0 + line_dir);
    Eigen::Vector3d leq3 = Xc1.cross(Xc2);
    leq3 = leq3 / leq3.segment<2>(0).norm();
    double err1 = fabs(xs.dot(leq3.segment<2>(0))+leq3(2));
    double err2 = fabs(xe.dot(leq3.segment<2>(0))+leq3(2));
    double se = err1 + err2;
    return se;
}

void vgl::ReprojectEndpointTo3DProjMat(const PMatrix& P, const Eigen::Vector2d& endpoint, const Eigen::Vector3d& X0,
                                       const Eigen::Vector3d& line_dir, Eigen::Vector3d* endpoint_3d)
{
    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = P.block<3,3>(0,0)*line_dir;
    Eigen::Vector3d b = -P.block<3,3>(0,0)*X0 - P.block<3,1>(0,3);
    A.col(1).segment<2>(0) = endpoint;
    A(2,1) = 1.0;
    Eigen::Vector2d x = A.colPivHouseholderQr().solve(b);
    *endpoint_3d = X0 + x(0)*line_dir;
}

void vgl::ReprojectEndpointTo3D(const Eigen::Vector3d& endpoint, const Eigen::Vector3d& X0, const Eigen::Vector3d& d_rot, double* p)
{
    Eigen::Matrix<double, 3, 2> M;
    M.col(1) = - d_rot;
    M.col(0) = endpoint;
    Eigen::Matrix<double, 2, 1> sol = M.colPivHouseholderQr().solve(X0);
    *p = sol(1);
}

void vgl::LineFrom3DEndpoints(const Eigen::Vector3d& X1, const Eigen::Vector3d& X2, Eigen::Vector3d* X0, Eigen::Vector3d* line_dir)
{
    *line_dir = X2-X1;
    *line_dir = *line_dir / (line_dir->norm());
    double a = (X1.dot(*line_dir));
    *X0 = X1 - a*(*line_dir);
}

void vgl::NormalizedLineEquation(double sx, double sy, double ex, double ey, const Eigen::Matrix3d& K, Eigen::Vector3d* lineEq)
{
    Eigen::Vector3d Xs(sx, sy, 1.0);
    Eigen::Vector3d Xe(ex, ey, 1.0);
    Eigen::Vector3d lineImg = Xs.cross(Xe);
    *lineEq = K.transpose() * lineImg;
    *lineEq = *lineEq / lineEq->segment<2>(0).norm();
}

Eigen::Vector3d vgl::MapPoint(const Eigen::Matrix4d& T_c2w, const Eigen::Vector3d& X)
{
    return T_c2w.block<3,3>(0,0).transpose() * (X - T_c2w.block<3,1>(0,3));
}

Eigen::Vector2d vgl::ProjectPoint(const Eigen::Matrix3d& K, Eigen::Vector3d& X)
{
    Eigen::Vector3d x_h = K*X;
    return x_h.segment<2>(0)/x_h(2);
}

void vgl::EncodeLineMinimal(const Eigen::Vector3d& X0, const Eigen::Vector3d& line_dir, Eigen::Matrix3d* R_line_p, double* alpha)
{
    Eigen::Matrix3d R_line;
    R_line.col(0) = line_dir;
    if (X0.norm() == 0)
    {
        Eigen::Vector3d dir1(1,0,0);
        Eigen::Vector3d dir2(0,1,0);
        Eigen::Vector3d X_help;
        if (fabs(dir1.dot(line_dir)) > fabs(dir2.dot(line_dir)))
        {
            X_help = line_dir.cross(dir2);
        } else {
            X_help = line_dir.cross(dir1);
        }
        X_help = X_help/X_help.norm();
        R_line.col(1) = X_help;
        R_line.col(2) = line_dir.cross(X_help);
    } else {
        R_line.col(1) = X0 / X0.norm();
        R_line.col(2) = line_dir.cross(X0) / X0.norm();
    }

    *R_line_p = R_line;
    *alpha = X0.norm();
}


void vgl::DecodeLineMinimal(const Eigen::Matrix3d& R, double alpha, Eigen::Vector3d* X0, Eigen::Vector3d* line_dir)
{
    *line_dir = R.col(0);
    *X0 = alpha * R.col(1);
}

Eigen::Matrix3d vgl::cpmat(const Eigen::Vector3d& t)
{
    Eigen::Matrix3d t_hat;
    t_hat << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;
    return t_hat;
}

bool vgl::TriangulateLineAsDepths(const vec2d& epts, const Eigen::Vector3d& l, const Eigen::Matrix3d& Sigma_l,
                             const vecmat2d& Sigmas_epts, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                             std::vector<double>* depths, std::vector<Eigen::Matrix3d>* covs, Eigen::Matrix3d* CrossCov)
{
//    std::vector<Eigen::Vector3d> gls;
//    for (int i = 0; i < 2; i++)
//    {
//
//        Eigen::Vector3d ept_h;
//        ept_h(2) = 1.0;
//        ept_h.segment<2>(0) = epts[i];
//
//        Eigen::Matrix3d Sigma_ept_h;
//        Sigma_ept_h.setZero();
//        Sigma_ept_h.block<2,2>(0,0) = Sigmas_epts[i];
//
//        double depth = -l.dot(t)/l.dot(R*ept_h);
//
//        double denom = l.dot(R * ept_h);
//        Eigen::Vector3d g_l = -t/denom + 1.0/denom/denom*R * ept_h;
//        Eigen::Vector3d g_ept_h =  l.dot(t)/denom/denom*R.transpose()*l;
//        double sigma_depth_2 = g_l.transpose() * Sigma_l * g_l + g_ept_h.transpose()*Sigma_ept_h*g_ept_h;
//
//        Eigen::Vector3d covDepthEpt_h = Sigma_ept_h * g_ept_h;
//
//        Eigen::Matrix3d CovX = sigma_depth_2 * ept_h * ept_h.transpose() +
//                Sigma_ept_h * depth * depth + covDepthEpt_h*depth*ept_h.transpose() + ept_h * depth * CovDepthEpt_h.transpose();
//        covs->push_back(CovX);
//        gls.push_back(g_l);
//    }
//    *CrossCov = gls[0] * Sigma_l * gls[1].transpose();
}