
#ifndef VO_VO_H
#define VO_VO_H

#include <vector>
#include <boost/circular_buffer.hpp>

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

class VisualOdometry {
public:
    VisualOdometry(const cv::Mat &K, size_t min_tracked_points);

    void addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman);

    cv::Mat drawMatches(const cv::Mat &image);

    cv::Mat draw3D();

private:

    const size_t kFrameBufferCapacity = 3;
    const int kBundleWindow = 20;

    FeatureDetector feature_detector_;
    FeatureTracker feature_tracker_;

    bool tracking_;
    size_t  min_tracked_points_;

    boost::circular_buffer<VOFrame> frame_buffer_;

    cv::Scalar color_;
    cv::Mat K_;

    cv::Mat last_keyframe_t_;

    KalmanFilter kf_;
    int count_ = 0;
    BundleAdjustment bundle_adjustment_;
};


#endif //VO_VO_H
