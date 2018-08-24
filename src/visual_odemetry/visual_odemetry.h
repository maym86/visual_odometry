

#ifndef VO_VO_H
#define VO_VO_H

#include <vector>

#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#include "src/kalman_filter/kalman_filter.h"

#include "vo_frame.h"

class VisualOdemetry {
public:
    VisualOdemetry(double focal, const cv::Point2d &pp);

    void addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman);

    cv::Mat drawMatches(const cv::Mat &image);

private:
    void triangulate(VOFrame *prev, VOFrame *now);
    double getScale(const VOFrame &prev, const VOFrame &now,  int num_points);

    const size_t kScale = 1;
    const size_t kMinTrackedPoints = 1500;

    FeatureDetector feature_detector_;
    FeatureTracker feature_tracker_;

    bool tracking_;

    VOFrame now_;
    VOFrame prev_;

    cv::Scalar color_;

    double focal_;
    cv::Point2d pp_;


    KalmanFilter kf_;
};


#endif //VO_VO_H
