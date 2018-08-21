

#ifndef VO_VO_H
#define VO_VO_H

#include <vector>

#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"

class VisualOdemetry {
public:
    VisualOdemetry(double focal, const cv::Point2d &pp);

    void addImage(const cv::Mat &image, cv::Mat *pose_R, cv::Mat *pose_t);

    cv::Mat drawMatches(const cv::Mat &image);

private:

    const size_t kMinTrackedPoints = 1500;
    const float kScale = 1;


    FeatureDetector feature_detector_;
    FeatureTracker feature_tracker_;

    bool tracking_;
    cv::Mat_<double> pose_t_;
    cv::Mat_<double> pose_R_;
    cv::Mat mask_;

    cv::cuda::GpuMat gpu_image_;
    cv::cuda::GpuMat prev_gpu_image_;
    std::vector<cv::Point2f> points_previous_;
    std::vector<cv::Point2f> points_;

    cv::Scalar color_;

    double focal_;
    cv::Point2d pp_;
};


#endif //VO_VO_H
