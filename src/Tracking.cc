/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*
* (C) 2018 Alexander Vakhitov <alexander.vakhitov at gmail dot com>
* Created: functions TrackReferenceSego and TriangulateLRPoints
* Modified: a function Track to use TrackReferenceSego(),
* the constructor to store SEGO properties,
*
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"
#include "PnPUsolver.h"

#include<iostream>

#include<mutex>
#include <opencv/cxeigen.hpp>
#include <include/SEGOLoop.h>


#include "Sleep.h"

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                   KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    bSegoRecovery = (int)fSettings["SEGORecovery"];

    mnPnpMode = (int)fSettings["PnPmode"];

    cout << "- Use stereo egomotion recovery: " << bSegoRecovery << endl;
    cout << "- Use PnP mode: " << mnPnpMode << endl;

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

PnPUsolver2* Tracking::SetupPnPU(const Frame &F, const vector<MapPoint*> &vpMapPointMatches)
{
    std::vector<Eigen::Matrix3d> sigmas3d;
    std::vector<cv::Point2f> p2D;
    std::vector<float> sigma2;

    std::vector<cv::Point3f> p3D;
    std::vector<size_t> keyPointIndices;
    std::vector<size_t> allIndices;

    int idx=0;
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if(pMP)
        {
            if(!pMP->isBad())
            {

                cv::Mat S3d = pMP->GetWorldCov();

//                    std::cout << " obtained sigma3d " << std::endl;
                if (S3d.cols == 0)
                {
                    continue;
//                    S3d = cv::Mat::zeros(3, 3, CV_32F);
                }
                Eigen::Matrix3d Sigma3d_eig;
                cv::cv2eigen(S3d, Sigma3d_eig);
                if (Sigma3d_eig.determinant() == 0)
                {
                    continue;
                }

                sigmas3d.push_back(Sigma3d_eig);

                const cv::KeyPoint &kp = F.mvKeysUn[i];

                p2D.push_back(kp.pt);
                sigma2.push_back(F.mvLevelSigma2[kp.octave]);

                cv::Mat Pos = pMP->GetWorldPos();
                p3D.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));

                keyPointIndices.push_back(i);
                allIndices.push_back(idx);

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    double fu = F.fx;
    double fv = F.fy;
    double uc = F.cx;
    double vc = F.cy;

    int n_map_points = vpMapPointMatches.size();

//    PnPUsolver2(const std::vector<Eigen::Matrix3d>& sigmas3d, const std::vector<cv::Point2f>& p2D,
//    const std::vector<float>& sigma2, const std::vector<cv::Point3f>& p3D,
//    const std::vector<size_t>& keyPointIndices, const std::vector<size_t>& allIndices,
//    double fu, double fv, double uc, double vc);
    PnPUsolver2* pnpUSolver = new PnPUsolver2(sigmas3d, p2D, sigma2, p3D, keyPointIndices, allIndices, fu, fv, uc, vc,
            n_map_points);
    return pnpUSolver;
}

PnPsolver* Tracking::SetupPnP(const Frame &F, const vector<MapPoint*> &vpMapPointMatches)
{
    int mode = mnPnpMode;

    std::vector<float> sigmas_3d;
    std::vector<Eigen::Matrix3d> sigmas_3d_full;
    std::vector<cv::Point2f> p2D;
    std::vector<float> sigma2;

    std::vector<cv::Point3f> p3D;
    std::vector<size_t> keyPointIndices;
    std::vector<size_t> allIndices;

    int idx=0;
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if(pMP)
        {
            if(!pMP->isBad())
            {

                cv::Mat S3d = pMP->GetWorldCov();

//                    std::cout << " obtained sigma3d " << std::endl;
                if (S3d.empty())
                {
                    continue;
//                    S3d = cv::Mat::zeros(3, 3, CV_32F);
                }
                Eigen::Matrix3d Sigma3d_eig;
                cv::cv2eigen(S3d, Sigma3d_eig);
                if (Sigma3d_eig.determinant() == 0)
                {
                    continue;
                }

                sigmas_3d_full.push_back(Sigma3d_eig);
                sigmas_3d.push_back(Sigma3d_eig.trace()*0.33333);

                const cv::KeyPoint &kp = F.mvKeysUn[i];

                p2D.push_back(kp.pt);
                sigma2.push_back(F.mvLevelSigma2[kp.octave]);

                cv::Mat Pos = pMP->GetWorldPos();
                p3D.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));

                keyPointIndices.push_back(i);
                allIndices.push_back(idx);

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    double fu = F.fx;
    double fv = F.fy;
    double uc = F.cx;
    double vc = F.cy;

    int n_map_points = vpMapPointMatches.size();

//    PnPsolver::PnPsolver(const std::vector<cv::Point2f> &p2D,
//    const std::vector<float> &sigma2, const std::vector<cv::Point3f> &p3D,
//    const std::vector<size_t> &keyPointIndices, const std::vector<size_t> &allIndices,
//    double fu, double fv, double uc, double vc, int nMapPoints,
//    const std::vector<float> &sigmas_3d,
//    const std::vector<Eigen::Matrix3d> &sigmas_3d_full,
//    int mode) :
    PnPsolver* pnpSolver = new PnPsolver(p2D, sigma2, p3D, keyPointIndices, allIndices, fu, fv, uc, vc,
                                              n_map_points,
                                              sigmas_3d,
                                              sigmas_3d_full,
                                              mode);
    return pnpSolver;
}


void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK) {
                        bOK = TrackReferenceSego();
                    }
                }
            }
            else
            {
                bOK = Relocalization();
                if (!bOK && bSegoRecovery)
                {
                    bOK = TrackReferenceSego();
                    std::cout << " tracking failed; localization failed; track sego " << bOK << std::endl;
                }
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK) {
                bOK = TrackLocalMap();
                if (!bOK && bSegoRecovery)
                {
                    std::cout << " track local map failed " << std::endl;
                    bOK = TrackReferenceSego();
                    std::cout << " sego tried " << bOK << std::endl;
                    bOK = TrackLocalMap();
                    std::cout << " track map  " << bOK << std::endl;
                }
            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

    void TriangulateLRPoints(const std::vector<std::vector<cv::KeyPoint>>& pt_trips, const Eigen::Vector3d& t12, const Eigen::Matrix3d& K,
                             double pix_thr, std::vector<Eigen::Vector3d>* pts3d_p, std::vector<std::vector<cv::KeyPoint>>* pt_trips_fin_p, std::vector<int>* wrong_lr_p)
    {
        cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64FC1);
        cv::Mat eye_mat = cv::Mat::eye(3, 3, CV_64FC1);
        eye_mat.copyTo(P0(cv::Rect(0, 0, 3, 3)));
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64FC1);
        eye_mat.copyTo(P1(cv::Rect(0, 0, 3, 3)));
        cv::Mat t2_cv;
        cv::eigen2cv(t12, t2_cv);
        t2_cv.copyTo(P1(cv::Rect(3, 0, 1, 3)));

        cv::Mat K_cv;
        cv::eigen2cv(K, K_cv);
        P0 = K_cv * P0;
        P1 = K_cv * P1;

        std::vector<cv::Point2d> pts_1, pts_2;
        for (size_t i = 0; i < pt_trips.size(); i++) {
            cv::Point2d pt1(pt_trips[i][0].pt);
            cv::Point2d pt2(pt_trips[i][1].pt);
            pts_1.push_back(pt1);
            pts_2.push_back(pt2);
        }
        cv::Mat pts_3d;
//    std::cout << " before triang" << std::endl;
        cv::triangulatePoints(P0, P1, pts_1, pts_2, pts_3d);
        std::vector<Eigen::Vector3d> pts3d;
        std::vector<std::vector<cv::KeyPoint>> point_triplets_fin;
        wrong_lr_p->clear();
        for (size_t i = 0; i < pt_trips.size(); i++)
        {
            Eigen::Vector3d X;
            X << pts_3d.at<double>(0, i), pts_3d.at<double>(1, i), pts_3d.at<double>(2, i);
            X = X / pts_3d.at<double>(3, i);
            Eigen::Vector3d Xc1 = X;
            Xc1 = K * Xc1/ Xc1(2);
            Eigen::Vector3d Xc2 = X + t12;
            Xc2 = K * Xc2/Xc2(2);
            Eigen::Vector3d Xc1_pred;
            Xc1_pred << pt_trips[i][0].pt.x, pt_trips[i][0].pt.y, 1.0;
            Eigen::Vector3d Xc2_pred;
            Xc2_pred << pt_trips[i][1].pt.x, pt_trips[i][1].pt.y, 1.0;
            double pe1 = (Xc1-Xc1_pred).norm();
            double pe2 = (Xc2-Xc2_pred).norm();
            if (isnanf(X.norm()) || X.norm() > 1e5)
            {
                continue;
            }
            if (pe1 < pix_thr && pe2 < pix_thr)
            {
                point_triplets_fin.push_back(pt_trips[i]);
                pts3d.push_back(X);
            } else {
                wrong_lr_p->push_back(i);
            }
        }
        *pts3d_p = pts3d;
        *pt_trips_fin_p = point_triplets_fin;
    }

    bool Tracking::TrackReferenceSego()
    {

        if (!bSegoRecovery)
        {
            return Tracking::TrackReferenceKeyFrame();
        }

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7,true);
        vector<MapPoint*> vpMapPointMatches, vpMapPointMatchesR;
        std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> kps, kpsR;
        int nmatches = matcher.SearchByBoWKP(mpReferenceKF,mCurrentFrame,kps,vpMapPointMatches,false);
        int nmatches_right = matcher.SearchByBoWKP(mpReferenceKF,mCurrentFrame,kpsR,vpMapPointMatchesR,true);

        if(nmatches+nmatches_right<7) {
            std::cout << " low No. of matches " << nmatches+nmatches_right << std::endl;
            return false;
        }

        //we form 4 sets of triplet matches between left-right-left  and right-left-right camera views,
        // and then we run SEGOLoop/P3PLoop to estimate camera motion using RANSAC
        cv::Mat Tcw;
        Eigen::Matrix3d K;
        cv::cv2eigen(mCurrentFrame.mK, K);
        double thr = 10;
        Eigen::Vector3d b(-mCurrentFrame.mb, 0, 0);

        //we form left-right-left (0,1,2) matches
        std::vector<std::vector<cv::KeyPoint>> set_lrl, set_lrl_fin;
        for (size_t i = 0; i < vpMapPointMatches.size(); i++)
        {
            if (vpMapPointMatches[i])
            {
                std::vector<cv::KeyPoint> kp_vec{kps[i].first, kps[i].second, mCurrentFrame.mvKeys[i]};
                set_lrl.push_back(kp_vec);
            }
        }
        //we triangulate the features and filter out those not located well
        std::vector<Eigen::Vector3d> pts3d_lrl;
        std::vector<int> wrong_lr;
        TriangulateLRPoints(set_lrl, b, K,thr, &pts3d_lrl, &set_lrl_fin, &wrong_lr);

        //we form right-left-right (1,0,3) matches
        std::vector<std::vector<cv::KeyPoint>> set_rlr, set_rlr_fin;
        for (size_t i = 0; i < vpMapPointMatchesR.size(); i++)
        {
            if (vpMapPointMatchesR[i])
            {
                std::vector<cv::KeyPoint> kp_vec{kpsR[i].second, kpsR[i].first, mCurrentFrame.mvKeysRight[i]};
                set_rlr.push_back(kp_vec);
            }
        }
        //we triangulate the features and filter out those not located well
        std::vector<Eigen::Vector3d> pts3d_rlr;
        std::vector<int> wrong_rl;
        TriangulateLRPoints(set_rlr, -b, K,thr, &pts3d_rlr, &set_rlr_fin, &wrong_rl);

        std::vector<std::vector<std::vector<cv::KeyPoint>>> pt_trips_sets{set_lrl_fin, set_rlr_fin, std::vector<std::vector<cv::KeyPoint>>(), std::vector<std::vector<cv::KeyPoint>>()};
        std::vector<std::vector<Eigen::Vector3d>> pts3d_for_trips{pts3d_lrl, pts3d_rlr, std::vector<Eigen::Vector3d>(), std::vector<Eigen::Vector3d>()};


        std::vector<std::vector<int>> tri_inds{std::vector<int>{0,1,2}, std::vector<int>{1,0,3},
                                               std::vector<int>{2,3,0}, std::vector<int>{3,2,1}};
        RANSACLoop* sego = new SEGOLoop(pt_trips_sets, 0.99, 0.75, K, pts3d_for_trips, thr, tri_inds, b);
        sego->Iterate(cv::Mat());
        //we decode and store the inliers
        if (sego->inlier_cnt > 5)
        {
            Eigen::Matrix<float, 4, 4> Tf;
            Tf.setIdentity();
            Tf.block<3, 3>(0, 0) = sego->R_best.cast<float>();
            Tf.block<3, 1>(0, 3) = sego->t_best.cast<float>();
            cv::eigen2cv(Tf, Tcw);
            mCurrentFrame.mvpMapPoints = vpMapPointMatches;
            std::map<int, bool> isPointUsed;
            for (size_t i = 0; i < vpMapPointMatches.size(); i++)
            {
                isPointUsed[i] = true;
            }
            for (int wi: wrong_lr)
            {
                isPointUsed[wi] = false;
            }
            int loc_ind = 0;
            for (size_t i = 0; i < vpMapPointMatches.size(); i++)
            {
                if (isPointUsed[i])
                {
                    loc_ind++;
                }
            }
            std::map<int, int> loc2glob_id;
            int gi = 0;
            for (size_t li = 0; li < set_lrl.size(); li++)
            {
                while (!isPointUsed[gi])
                {
                    gi++;
                }
                loc2glob_id[li] = gi;
                gi++;
            }
            int inlier_cnt = 0;
            for (size_t li = 0; li < set_lrl.size(); li++)
            {
                int gii = loc2glob_id[li];
                if (sego->is_inlier[li])
                {
                    mCurrentFrame.mvpMapPoints[gii] = vpMapPointMatches[gii];
                    inlier_cnt++;
                } else {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[gii];
                    if (!pMP)
                    {
                        continue;
                    }
                    mCurrentFrame.mvpMapPoints[gii] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[gii]=false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                }
            }
        } else {
            mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        }

        if (Tcw.empty())
        {
            std::cout << " sego failed " << std::endl;
            Tcw = cv::Mat();
        }

        if (!Tcw.empty())
        {
            mCurrentFrame.SetPose(Tcw * mpReferenceKF->GetPose());


            ORBmatcher matcher(0.9,true);

            // Update last frame pose according to its reference keyframe
            // Create "visual odometry" points if in Localization Mode
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

            // Project points seen in previous frame
            int th;
            if(mSensor!=System::STEREO)
                th=15;
            else
                th=7;
            int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

            // If few matches, uses a wider window search
            if(nmatches<20)
            {
                fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
                nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
            }

        } else {
            mCurrentFrame.mvpMapPoints = vpMapPointMatches;
            if (!mLastFrame.mTcw.empty()) {
                mCurrentFrame.SetPose(mLastFrame.mTcw);
            } else {
                return false;
            }

        }



        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }

        return nmatchesMap>=10;
    }





bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

    typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> posevector;

    void RunEPnPU(const Eigen::Matrix3d& Re, const Eigen::Vector3d& te, const std::vector<cv::KeyPoint>& keypts,
                  const std::vector<float>& sigma_levels_2, const std::vector<Eigen::Vector3d>& XX,
                  const std::vector<Eigen::Matrix3d>& Sigmas3dGlobal, const Eigen::Matrix3d& K, int mode,
                  posevector* sols_epnp_p)
    {
        Eigen::Vector3d Xm;
        Xm.setZero();
//    for (int i = 0; i < XX.size(); i++)
//    {
//        Xm = Xm + XX[i];
//    }
//    Xm = Xm / XX.size();
//
        int n_point = keypts.size();
        cv::Mat p3d(n_point, 3, CV_32FC1);
        for (int i = 0; i < n_point; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                p3d.at<float>(i, j) = XX[i](j) - Xm(j);
            }

        }

        for (int i = 0; i < Sigmas3dGlobal.size(); i++)
        {
            if (isnanf(Sigmas3dGlobal[i].determinant()))
            {
                std::cout <<"nan sigma before pnpu " << std::endl;
            }
        }

        ORB_SLAM2::PnPUsolver pnp_solver(keypts, sigma_levels_2, p3d, Sigmas3dGlobal, K(0, 0), K(1, 1), K(0, 2), K(1, 2), mode);
        Eigen::Matrix3d R_est;
        Eigen::Vector3d t_est;
        if (pnp_solver.SolveForAll(Re, te, &R_est, &t_est))
        {
            std::cout << " solved " << Re << " " << R_est << std::endl;
            Eigen::Matrix4d T;
            T.setIdentity();
            T.block<3, 3>(0, 0) = R_est;
            T.block<3, 1>(0, 3) = t_est-R_est * Xm;
            sols_epnp_p->push_back(T);
        } else {
            std::cout << " not solved " << std::endl;
        }
    }

    typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vec2d;

//    int FindBestSolutionReproj(const std::vector<Eigen::Vector3d>& XX, const vec2d& xx, const Eigen::Matrix3d& K,
//                               const posevector& poses, double* min_err_p)
//    {
//        double min_err = 1e100;
//        int sol_ind = -1;
//        for (int i = 0; i < poses.size(); i++)
//        {
//            auto T = poses[i];
//            double err = 0;
//            for (int j = 0; j < XX.size(); j++)
//            {
//                Eigen::Vector3d xc = K*(T.block<3,3>(0,0) * XX[i] + T.block<3,1>(0,3));
//                xc = xc / xc(2);
//                err += (xc.segment<2>(0) - xx[i]).norm();
//            }
//            if (err < min_err)
//            {
//                min_err = err;
//                sol_ind = i;
//            }
//        }
//        *min_err_p = min_err / XX.size();
//        return sol_ind;
//    }

bool Tracking::Relocalization()
{
    std::cout << " Reloc run " << std::endl;
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<PnPUsolver2*> vpPnPUsolvers;
    vpPnPUsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver *pSolver = SetupPnP(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;

            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            cv::Mat Tcw;

            PnPsolver* pSolver = vpPnPsolvers[i];
            Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            std::cout << " pnpsolver output " << Tcw << std::endl;

            if (bUseUncertainty && false)
            {
//                void RunEPnPU(const Eigen::Matrix3d& Re, const Eigen::Vector3d& te, const std::vector<cv::KeyPoint>& keypts,
//                              const std::vector<float>& sigma_levels_2, const std::vector<Eigen::Vector3d>& XX,
//                              const std::vector<Eigen::Matrix3d>& Sigmas3dGlobal, const Eigen::Matrix3d& K, int mode,
//                              posevector* sols_epnp_p)
//                Eigen::Matrix4d T_eig;
//                if (Tcw.cols > 0)
//                {
//                    cv::cv2eigen(Tcw, T_eig);
//                } else {
//                    T_eig.setIdentity();
//                }
//
//                Eigen::Matrix3d Re = T_eig.block<3,3>(0,0);
//                Eigen::Vector3d te = T_eig.block<3,1>(0,3);
//                std::vector<cv::KeyPoint> keypts;
//                std::vector<float> sigma_levels_2;
//                std::vector<Eigen::Vector3d> XX;
//                std::vector<Eigen::Matrix3d> Sigmas3dGlobal;
//                vec2d xx;
//                for (auto pMP: vvpMapPointMatches[i])
//                {
//                    if(pMP)
//                    {
//                        if(!pMP->isBad())
//                        {
//
//                            cv::Mat S3d = pMP->GetWorldCov();
//
//
//                            if (S3d.cols == 0)
//                            {
//                                continue;
////                    S3d = cv::Mat::zeros(3, 3, CV_32F);
//                            }
//                            Eigen::Matrix3d Sigma3d_eig;
//                            cv::cv2eigen(S3d, Sigma3d_eig);
//                            if (Sigma3d_eig.determinant() == 0)
//                            {
//                                continue;
//                            }
//
//                            std::cout << " obtained sigma3d " << S3d << std::endl;
//
//                            Sigmas3dGlobal.push_back(Sigma3d_eig);
//
//                            const cv::KeyPoint &kp = mCurrentFrame.mvKeysUn[i];
//
//                            keypts.push_back(kp);
//
//                            cv::Mat Pos = pMP->GetWorldPos();
//                            XX.push_back(Eigen::Vector3d(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));
//                            xx.push_back(Eigen::Vector2d(kp.pt.x, kp.pt.y));
//                        }
//                    }
//
//                }
//                posevector sols_epnp;
//                Eigen::Matrix3d K;
//                K.setIdentity();
//                K(0,0) = mCurrentFrame.fx;
//                K(1,1) = mCurrentFrame.fy;
//                K(0,2) = mCurrentFrame.cx;
//                K(1,2) = mCurrentFrame.cy;
//                int mode = 2;//mfull
//                for (int oi = 0; oi < 8; oi++)
//                {
//                    sigma_levels_2.push_back(mCurrentFrame.mvLevelSigma2[oi]);
//                }
//                RunEPnPU(Re, te, keypts, sigma_levels_2, XX, Sigmas3dGlobal, K, mode, &sols_epnp);
//                double min_err_epnp;
//                int sol_ind = FindBestSolutionReproj(XX, xx, K, sols_epnp, &min_err_epnp);
//
//
//
//                if (sols_epnp.size() > 0 && sol_ind >= 0)
//                {
//                    for (int ri = 0; ri < 3; ri++ )
//                    {
//                        for (int ci = 0; ci < 4; ci++)
//                        {
//                            Tcw.at<float>(ri, ci) = sols_epnp[sol_ind](ri, ci);
//                        }
//                    }
//                }
//
//                std::cout << " inliers " << XX.size() << std::endl;
//                std::cout << sol_ind << " " <<std::endl;
//                if (sol_ind >= 0)
//                {
//                    std::cout << sols_epnp[sol_ind] << std::endl;
//                }
//                std::cout << " solution " << Tcw << std::endl;

            }


            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
//                std::cout << " found a pose " << std::endl;

                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    std::cout << " reloc successful " << std::endl;
                    bMatch = true;
                    break;
                }
            }
        }
    }



    if(!bMatch)
    {
        std::cout << " reloc unsuccessful " << std::endl;
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            sleep_ms(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
    mVelocity = cv::Mat();
}



} //namespace ORB_SLAM
