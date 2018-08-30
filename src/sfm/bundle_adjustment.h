

#ifndef VO_SFM_BUNDLE_ADJUSTMENT_H
#define VO_SFM_BUNDLE_ADJUSTMENT_H

#include <vector>
#include <opencv2/features2d.hpp>

#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include "src/visual_odemetry/vo_frame.h"

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#else
#include "src/features/feature_detector.h"
#endif

class BundleAdjustment {

public:
    void init(size_t max_frames);
    void addKeyFrame(const VOFrame &frame, float focal, cv::Point2d pp, int feature_count=0);
    int slove(cv::Mat *R, cv::Mat *t);

private:

    FeatureDetector feature_detector_;

    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster_;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher_;


    std::vector<cv::detail::CameraParams> cameras_;
    std::vector<cv::detail::ImageFeatures> features_;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_;

    size_t max_frames_;
    int count_ = 0;
};


#endif //VO_SFM_BUNDLE_ADJUSTMENT_H
