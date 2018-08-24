
#include "feature_tracker.h"

FeatureTracker::FeatureTracker(){
     optical_flow_ = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
}


void FeatureTracker::trackPoints(VOFrame *prev, VOFrame *now) {
    cv::cuda::GpuMat now_points_gpu;
    cv::cuda::GpuMat status_gpu;

    //Copy previous data to GPU memory
    cv::cuda::GpuMat prev_points_gpu;
    cv::Mat prev_points_mat(1, (int) prev->points.size(), CV_32FC2, (void*) &(prev->points)[0]);
    prev_points_gpu.upload(prev_points_mat);

    optical_flow_->calc(prev->gpu_image, now->gpu_image, prev_points_gpu, now_points_gpu, status_gpu);

    //Get results back from GPU memory
    now->points.resize(now_points_gpu.cols);
    cv::Mat now_points_mat(1, now_points_gpu.cols, CV_32FC2, (void*) &(now->points)[0]);
    now_points_gpu.download(now_points_mat);

    std::vector<unsigned char> status(status_gpu.cols);
    cv::Mat status_mat(1, status_gpu.cols, CV_8UC1, (void*)&status[0]);
    status_gpu.download(status_mat);


    //Remove bad points
    for (int i = status.size() - 1; i >= 0; --i) {
        cv::Point2f pt = now->points[i];
        if (status[i] == 0 || pt.x < 0 || pt.y < 0) {
            if (pt.x < 0 || pt.y < 0) {
                status[i] = 0;
            }
            prev->points.erase(prev->points.begin() + i);
            now->points.erase(now->points.begin() + i);
        }
        else{
            prev->tracked_index.push_back(i); //indices of points that are kept
        }
    }
    std::reverse(prev->tracked_index.begin(),prev->tracked_index.end());
}

