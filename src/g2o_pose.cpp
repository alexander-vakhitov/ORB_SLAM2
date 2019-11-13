//
// Created by alexander on 05.11.19.
//

#include "g2o_pose.h"

G2O_Pose::G2O_Pose(int refType, double _delta, const Eigen::Matrix3d& _K, const Eigen::Matrix4d &_T_init, bool is_debug,
        std::string debug_path) :
        refinementType(refType), delta(_delta), K(_K), nInitialCorrespondences(0), T_init(_T_init), is_debug(is_debug), debug_path(debug_path)
{
}

int G2O_Pose::Refine(Eigen::Matrix4d *T_fin, const std::vector<cv::Point2f>& p2D,
const std::vector<float>& sigmas2D, const std::vector<cv::Point3f>& p3D, const std::vector<float>& sigmas3D,
const std::vector<Eigen::Matrix3d>& sigmas3DFull, const std::vector<bool>& vbInliers, double s3dmin, double s3dmax)
{
    g2o::SparseOptimizer optimizer;

    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set Frame vertex
    vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setId(0);
    vSE3->setFixed(false);
    g2o::SE3Quat se3_init(T_init.block<3,3>(0,0), T_init.block<3,1>(0,3));
    vSE3->setEstimate(se3_init);
    optimizer.addVertex(vSE3);

//    std::cout << " opt inited K = " << K << std::endl;

    if (is_debug)
    {
        debug_out = std::ofstream(debug_path);
    }

    for (int ii = 0; ii < p3D.size(); ii++) {
        if (vbInliers[ii])
        {
            AddCorrespondence(optimizer, ii+1, p3D[ii].x, p3D[ii].y, p3D[ii].z, p2D[ii].x, p2D[ii].y,
                                       sigmas2D[ii], sigmas3D[ii], sigmas3DFull[ii], vbInliers[ii],
                                       s3dmin, s3dmax);

        }
    }


    if (is_debug)
    {
        debug_out.close();
    }

//    std::cout << "opt started with " << edges.size() << " edges" << std::endl;

    T_fin->setIdentity();


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

    float chi2Mono[4]={5.991,5.991,5.991,5.991};
    float chi2Stereo[4]={7.815,7.815,7.815, 7.815};

//    for (int i = 0; i < 4; i++)
//    {
//        chi2Mono[i] = thrCoeff * chi2Mono[i];
//        chi2Stereo[i] = thrCoeff * chi2Stereo[i];
//    }

    const int its[4]={10,10,10,10};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        g2o::SE3Quat se3_init(T_init.block<3,3>(0,0), T_init.block<3,1>(0,3));
        vSE3->setEstimate(se3_init);


//        std::cout << "vertex count  " << optimizer.vertices().size() << std::endl;
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);
        nBad=0;
        for(size_t i=0, iend=edges.size(); i<iend; i++)
        {
            PoseEdge* e = edges[i];

//            const size_t idx = vnIndexEdgeMono[i];
//
            if(outlier_status[i])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                outlier_status[i] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
//                pFrame->mvbOutlier[idx]=false;
                outlier_status[i] = false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

//        std::cout << " num of edges " << optimizer.edges().size() << " bad edges " << nBad << std::endl;

        if(optimizer.edges().size()<10)
            break;
    }

    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    T_fin->setIdentity();
    T_fin->block<3,3>(0,0) = SE3quat_recov.rotation().toRotationMatrix();
    T_fin->block<3,1>(0,3) = SE3quat_recov.translation();

    return nInitialCorrespondences-nBad;
}

void G2O_Pose::AddCorrespondence(g2o::SparseOptimizer& optimizer, int id, double X, double Y, double Z, double u, double v, const double s2d, const double s3d,
                                 const Eigen::Matrix3d &S3D, bool is_inlier, double s3dmin, double s3dmax) {

    if (is_debug)
    {
        debug_out << X << " " << Y << " "<< Z << " "<< u << " "<< v << " "<< s2d << " "<< s3d << " ";
        for (int ii = 0; ii < 3; ii++)
        {
            for (int jj = 0; jj < 3; jj++)
            {
                debug_out << S3D(ii, jj) << " ";
            }
        }
        debug_out << int(is_inlier) << std::endl;
    }

    switch (refinementType)
    {
        case 0: {
            g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

            e->setId(id);

            e->fx = K(0, 0);
            e->fy = K(1, 1);
            e->cx = K(0, 2);
            e->cy = K(1, 2);

            e->Xw[0] = X;
            e->Xw[1] = Y;
            e->Xw[2] = Z;

            Eigen::Matrix2d Info = 1.0 / s2d * Eigen::Matrix2d::Identity();
            e->setInformation(Info);

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(Eigen::Vector2d(u, v));

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);

            if (!is_inlier)
            {
                e->setLevel(1);
            } else {
                e->setLevel(0);
            }
            rk->setDelta(delta);

            e->computeError();

//    std::cout << " added edge err " << eb->error() << std::endl;

            optimizer.addEdge(e);

            PoseEdge* eb = dynamic_cast<PoseEdge *>(e);
            edges.push_back(eb);

            break;
        }
        case 1: {
            g2o::EdgeSE3CovProjectXYZOnlyPose *ec = new g2o::EdgeSE3CovProjectXYZOnlyPose();

            ec->setId(id);

            ec->fx = K(0, 0);
            ec->fy = K(1, 1);
            ec->cx = K(0, 2);
            ec->cy = K(1, 2);

            ec->sigma_d_2 = s2d;



            if (s3dmin == -1) {
                ec->sigma_p_2 = s3d;
                ec->Sigma_p_2 = S3D; //Eigen::Matrix3d::Identity() * ec->sigma_p_2;//
            } else {
                double c = 1.0;
                if (s3d < s3dmin)
                {
                    c = s3dmin/s3d;
                }
                if (s3d > s3dmax)
                {
                    c = s3dmax/s3d;
                }
                ec->sigma_p_2 = c * s3d;
                ec->Sigma_p_2 = c * S3D; //Eigen::Matrix3d::Identity() * ec->sigma_p_2;//
            }

            ec->Xw[0] = X;
            ec->Xw[1] = Y;
            ec->Xw[2] = Z;
            Eigen::Matrix2d Info = Eigen::Matrix2d::Identity();
            ec->setInformation(Info);


            ec->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            ec->setMeasurement(Eigen::Vector2d(u, v));

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ec->setRobustKernel(rk);

            if (!is_inlier)
            {
                ec->setLevel(1);
            } else {
                ec->setLevel(0);
            }
            rk->setDelta(delta);

            ec->computeError();

//    std::cout << " added edge err " << eb->error() << std::endl;

            optimizer.addEdge(ec);

            PoseEdge* eb = dynamic_cast<PoseEdge *>(ec);
            edges.push_back(eb);
            break;
        }
    }



    outlier_status.push_back(!is_inlier);
    nInitialCorrespondences++;
}

void GetJProjOpt(double Xc, double Yc, double Zc, Eigen::Matrix<double, 2, 3>* J_proj_p)
{
    Eigen::Matrix<double, 2, 3> J_proj;
    J_proj.setZero();
    J_proj(0, 0) = 1.0 / Zc;
    J_proj(1, 1) = 1.0 / Zc;
    J_proj(0, 2) = -Xc / Zc / Zc;
    J_proj(1, 2) = -Yc / Zc / Zc;
    *J_proj_p = J_proj;
}

void GetJProjOptStereo(double Xc, double Yc, double Zc, double b, Eigen::Matrix3d* J_proj_p)
{
    Eigen::Matrix3d J_proj;
    J_proj.setZero();
    J_proj(0, 0) = 1.0 / Zc;
    J_proj(1, 1) = 1.0 / Zc;
    J_proj(2, 0) = 1.0 / Zc;
    J_proj(0, 2) = -Xc / Zc / Zc;
    J_proj(1, 2) = -Yc / Zc / Zc;
    J_proj(2, 2) = -(Xc-b) / Zc / Zc;
    *J_proj_p = J_proj;
}
