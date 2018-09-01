

#ifndef VO_SFM_BUNDLE_ADJUSTMENT_H
#define VO_SFM_BUNDLE_ADJUSTMENT_H

#include <vector>
#include <opencv2/features2d.hpp>

#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include "src/visual_odometry/vo_frame.h"

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#else
#include "src/features/feature_detector.h"
#endif


#include "pba/src/pba/pba.h"

class BundleAdjustment {

public:
    void init(size_t max_frames);
    void addKeyFrame(const VOFrame &frame, float focal, cv::Point2f pp);
    int slove(cv::Mat *R, cv::Mat *t);

private:

    void setPBAData(const std::vector<cv::detail::ImageFeatures> &features, const std::vector<cv::detail::MatchesInfo> &pairwise_matches, const std::vector<cv::Mat> &poses,
                    const cv::Point2f &pp, std::vector<Point3D> *pba_point_data, std::vector<Point2D> *pba_measurements, std::vector<int> *pba_camidx, std::vector<int> *pba_ptidx);

    FeatureDetector feature_detector_;

    cv::Ptr<cv::detail::FeaturesMatcher> matcher_;

    std::vector<CameraT> pba_cameras_;    //camera (input/ouput)
    std::vector<cv::Mat> poses_;
    std::vector<cv::detail::ImageFeatures> features_;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_;

    ParallelBA pba_;


    std::vector<Point3D>        pba_point_data_;     //3D point(iput/output)
    std::vector<Point2D>        pba_measurements_;   //measurment/projection vector
    std::vector<int>            pba_camidx_, pba_ptidx_;  //index of camera/point for each projection



    size_t max_frames_;
    int count_ = 0;
};


#endif //VO_SFM_BUNDLE_ADJUSTMENT_H
