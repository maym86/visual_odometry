
#ifndef VISUALODEMETRY_FETAURE_TRACKER_H
#define VISUALODEMETRY_FETAURE_TRACKER_H

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/cudaoptflow.hpp>

class FeatureTracker {
public:
    FeatureTracker();

    std::vector<cv::Point2f> trackPoints(const cv::cuda::GpuMat &img0, const cv::cuda::GpuMat &img1, std::vector<cv::Point2f> *prev_points);

private:
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optical_flow_;
};
#endif //VISUALODEMETRY_FETAURE_TRACKER_H
