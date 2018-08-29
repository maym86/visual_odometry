

#ifndef VO_SFM_BUNDLE_ADJUSTMENT_H
#define VO_SFM_BUNDLE_ADJUSTMENT_H

#include <vector>
#include <opencv2/features2d.hpp>

#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include "src/visual_odemetry/vo_frame.h"

class BundleAdjustment {
    explicit BundleAdjustment(size_t max_frames);
    void addKeyFrame(const VOFrame &frame, float focal, cv::Point2d pp);
    int slove(cv::Mat *R, cv::Mat *t);

private:
    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster_;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher_;


    std::vector<cv::detail::CameraParams> cameras_;
    std::vector<cv::detail::ImageFeatures> features_;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_;

    size_t max_frames_;
};


#endif //VO_SFM_BUNDLE_ADJUSTMENT_H
