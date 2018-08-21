
#include "feature_tracker.h"

FeatureTracker::FeatureTracker(){
     optical_flow_ = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
}


std::vector<cv::Point2f> FeatureTracker::trackPoints(const cv::cuda::GpuMat &img0, const cv::cuda::GpuMat &img1, std::vector<cv::Point2f> *prev_points) {
    cv::cuda::GpuMat next_points_gpu;
    cv::cuda::GpuMat status_gpu;

    cv::cuda::GpuMat prev_points_gpu;
    cv::Mat prev_points_mat(1, (int) prev_points->size(), CV_32FC2, (void*) &(*prev_points)[0]);
    prev_points_gpu.upload(prev_points_mat);

    optical_flow_->calc(img0, img1, prev_points_gpu, next_points_gpu, status_gpu);

    std::vector<cv::Point2f> next_points(next_points_gpu.cols);
    cv::Mat next_points_mat(1, next_points_gpu.cols, CV_32FC2, (void*)&next_points[0]);
    next_points_gpu.download(next_points_mat);

    std::vector<unsigned char> status(status_gpu.cols);
    cv::Mat status_mat(1, status_gpu.cols, CV_8UC1, (void*)&status[0]);
    status_gpu.download(status_mat);

    //Remove bad points
    for (int i = status.size() - 1; i >= 0; --i) {

        cv::Point2f pt = next_points[i];
        if (status[i] == 0 || pt.x < 0 || pt.y < 0) {
            if (pt.x < 0 || pt.y < 0) {
                status[i] = 0;
            }
            prev_points->erase(prev_points->begin() + i);
            next_points.erase(next_points.begin() + i);
        }
    }

    return next_points;
}

