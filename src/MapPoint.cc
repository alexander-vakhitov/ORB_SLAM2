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
*/

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>
#include <Eigen/Dense>
#include <opencv/cxeigen.hpp>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);

    mWorldCov = cv::Mat::zeros(3, 3, CV_32F);



    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);

    mWorldCov = cv::Mat::zeros(3, 3, CV_32F);

    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

//void MapPoint::SetWorldCov(const cv::Mat &Cov)
//{
//    unique_lock<mutex> lock2(mGlobalMutex);
//    unique_lock<mutex> lock(mMutexPos);
//    Cov.copyTo(mWorldCov);
//}

cv::Mat MapPoint::GetInformation(int type, const cv::Mat& R, const cv::Mat& t, double fx, double fy, double cx, double cy)
{
    unique_lock<mutex> lock(mMutexPos);
    cv::Mat Xc = R * mWorldPos + t;
    if (type == 0)
    {

    }
    return cv::Mat();
}


cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

void MapPoint::SetWorldCov(const cv::Mat& covMat) {
//    unique_lock<mutex> lock(mMutexPos);
//    mWorldCov = covMat.clone();
}

cv::Mat MapPoint::GetWorldCov(const Eigen::Vector3d& cam_center)
{
//    unique_lock<mutex> lock(mMutexPos);
//    if (mWorldCov.cols > 0)
//    {
//        return mWorldCov.clone();
//    } else {
//        return cv::Mat();
//    }

    std::pair<KeyFrame*, int> best_obs;
    best_obs.first = NULL;
    best_obs.second = -1;
    double best_dist = 1e6;
    for (auto& obs: mObservations)
    {
        KeyFrame* kf = obs.first;
        int id = obs.second;

        if (id < 0 || kf->mvDepth[id] < 0)
        {
            continue;
        }
        Eigen::Vector3d kfc;
        cv::cv2eigen(kf->GetCameraCenter(), kfc);
        double d = (cam_center - kfc).norm();
        if (kf->mvDepth[id] < best_dist)
        {
//            best_depth = kf->mvDepth[id];
            best_dist = d;
            best_obs = obs;
        }
    }
    if (best_obs.second >= 0)
    {
        KeyFrame* best_kf = best_obs.first;
        int best_id = best_obs.second;
        return best_kf->UnprojectPointCovFromParams(best_id, GetWorldPos());
    }
    return cv::Mat();
}

void GetMonoJac(const Eigen::Vector3d& Xc, Eigen::Matrix<double, 2, 3>* J_mono)
{
    J_mono->setZero();
    (*J_mono)(0,0) = 1.0/Xc(2);
    (*J_mono)(0,2) = -Xc(0)/Xc(2)/Xc(2);
    (*J_mono)(1,1) = 1.0/Xc(2);
    (*J_mono)(1,2) = -Xc(1)/Xc(2)/Xc(2);
}

void GetStereoJac(const Eigen::Vector3d& Xc, double b, Eigen::Matrix3d* J_s)
{
    J_s->setZero();
    (*J_s)(0,0) = 1.0/Xc(2);
    (*J_s)(0,2) = -Xc(0)/Xc(2)/Xc(2);
    (*J_s)(1,1) = 1.0/Xc(2);
    (*J_s)(1,2) = -Xc(1)/Xc(2)/Xc(2);
    (*J_s)(2,0) = 1.0/Xc(2);
    (*J_s)(2,2) = -(Xc(0)-b)/Xc(2)/Xc(2);
}

cv::Mat MapPoint::GetWorldCovFull()
{
    unique_lock<mutex> lock(mMutexFeatures);
    int cnt = 0;
    for (auto& obs: mObservations)
    {
        KeyFrame* kf = obs.first;
        int id = obs.second;

        if (id < 0)
        {
            continue;
        }
        if (kf->mvDepth[id] < 0)
        {
            cnt += 2;
        } else {
            cnt += 3;
        }
    }
    if (cnt < 3)
    {
        return cv::Mat();
    }

    cv::Mat X_cv = GetWorldPos();
    Eigen::Vector3d X;
    cv::cv2eigen(X_cv, X);

    Eigen::Matrix3d JtJ, JtJ_w;
    JtJ.setZero();
    JtJ_w.setZero();
    cnt = 0;
    for (auto& obs: mObservations)
    {
        KeyFrame* kf = obs.first;
        int id = obs.second;

        if (id < 0)
        {
            continue;
        }

        Eigen::Matrix4d Tcw_eig;
        cv::cv2eigen(kf->GetPose(), Tcw_eig);
        Eigen::Vector3d Xc = Tcw_eig.block<3,3>(0,0) * X + Tcw_eig.block<3,1>(0,3);

        if (kf->mvDepth[id] < 0)
        {
            Eigen::Matrix<double, 2, 3> J_mono;
            GetMonoJac(Xc, &J_mono);
            J_mono = J_mono * Tcw_eig.block<3,3>(0,0);
            JtJ = JtJ + J_mono.transpose() * J_mono;
            JtJ_w = JtJ_w + J_mono.transpose() * J_mono * kf->mvLevelSigma2[kf->mvKeys[id].octave] / kf->fx / kf->fx;
            cnt += 2;
        } else {
            Eigen::Matrix3d J_s;
            GetStereoJac(Xc, kf->mb, &J_s);
            J_s = J_s * Tcw_eig.block<3,3>(0,0);
            JtJ = JtJ + J_s.transpose() * J_s;
            JtJ_w = JtJ_w + J_s.transpose() * J_s* kf->mvLevelSigma2[kf->mvKeys[id].octave] / kf->fx / kf->fx;
            cnt += 3;
        }
    }
    Eigen::Matrix3d Sigma_eig = JtJ.inverse() * JtJ_w * JtJ.inverse();
//    std::cout << "sigma full " << std::endl;
//    std::cout << Sigma_eig << std::endl;
    cv::Mat Sigma(3, 3, CV_32FC1);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Sigma.at<float>(i, j) = Sigma_eig(i, j);
        }
    }
    return Sigma.clone();
}





cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF]=idx;

    if(pKF->mvuRight[idx]>=0)
        nObs+=2;
    else
        nObs++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF,mit->second);
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;
        pRefKF=mpRefKF;
        Pos = mWorldPos.clone();
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mNormalVector = normal/n;
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
