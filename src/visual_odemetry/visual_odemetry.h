

#ifndef VO_VO_H
#define VO_VO_H

#include <vector>

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#include "src/features/cuda/feature_tracker.h"
#else
#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#endif

#include "src/kalman_filter/kalman_filter.h"

#include "vo_frame.h"


class VisualOdemetry {
public:
    VisualOdemetry(double focal, const cv::Point2d &pp);

    void addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman);

    cv::Mat drawMatches(const cv::Mat &image);

private:
    const size_t kScale = 1;
    const size_t kMinTrackedPoints = 1500;

    FeatureDetector feature_detector_;
    FeatureTracker feature_tracker_;

    bool tracking_;

    //TODO make vo0 the current sate and so that we can update the pose if vo1 needs new detecion
    VOFrame vo2_;
    VOFrame vo1_;
    VOFrame vo0_;

    cv::Scalar color_;

    double focal_;
    cv::Point2d pp_;


    KalmanFilter kf_;
};


#endif //VO_VO_H
