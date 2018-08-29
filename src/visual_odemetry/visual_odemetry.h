

#ifndef VO_VO_H
#define VO_VO_H

#include <vector>
#include <boost/circular_buffer.hpp>
#include <src/sfm/bundle_adjustment.h>

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#include "src/features/cuda/feature_tracker.h"
#else
#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#endif

#include "src/kalman_filter/kalman_filter.h"
#include "src/sfm/bundle_adjustment.h"


#include "vo_frame.h"

class VisualOdemetry {
public:
    VisualOdemetry(double focal, const cv::Point2d &pp);

    void addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman);

    cv::Mat drawMatches(const cv::Mat &image);

private:
    const size_t kFrameBufferCapacity = 3;
    const size_t kMinTrackedPoints = 1500;
    const size_t kMinPosePoints = 8;

    FeatureDetector feature_detector_;
    FeatureTracker feature_tracker_;

    bool tracking_;

    boost::circular_buffer<VOFrame> frame_buffer_;

    cv::Scalar color_;

    double focal_;
    cv::Point2d pp_;

    cv::Mat last_keyframe_t_;

    KalmanFilter kf_;

    BundleAdjustment bundle_adjustment_;

};


#endif //VO_VO_H
