
#if __has_include("opencv2/cudaoptflow.hpp")

#include "feature_tracker.h"
#include "src/features/utils.h"

FeatureTracker::FeatureTracker(){
     gpu_optical_flow_ = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
}

void FeatureTracker::trackPointsGPU(VOFrame *vo0, VOFrame *vo1) {
    cv::cuda::GpuMat now_points_gpu;
    cv::cuda::GpuMat status_gpu;

    //Copy previous data to GPU memory
    cv::cuda::GpuMat prev_points_gpu;
    cv::Mat prev_points_mat(1, (int) vo0->points.size(), CV_32FC2, (void*) &(vo0->points)[0]);
    prev_points_gpu.upload(prev_points_mat);

    gpu_optical_flow_->calc(vo0->gpu_image, vo1->gpu_image, prev_points_gpu, now_points_gpu, status_gpu);

    //Get results back from GPU memory
    vo1->points.resize(now_points_gpu.cols);
    cv::Mat now_points_mat(1, now_points_gpu.cols, CV_32FC2, (void*) &(vo1->points)[0]);
    now_points_gpu.download(now_points_mat);

    std::vector<unsigned char> status(status_gpu.cols);
    cv::Mat status_mat(1, status_gpu.cols, CV_8UC1, (void*)&status[0]);
    status_gpu.download(status_mat);

    removePoints(&vo0, &vo1, &status);
}

#endif