
#ifndef VO_FEATURES_CUDA_DETECTION_H
#define VO_FEATURES_CUDA_DETECTION_H

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/cudafeatures2d.hpp>

#include "src/visual_odometry/vo_frame.h"

class FeatureDetector {

public:

    FeatureDetector();

    void detectFAST(VOFrame *frame);

    void detectFAST(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints);

    void detectComputeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);

    void computeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);

    void detectComputeAKAZE(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);
    void detectComputeBRISK(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);

private:
    const int kMaxFeatures = 10000;

    cv::Ptr<cv::cuda::FastFeatureDetector> gpu_detector_;

    cv::Ptr<cv::cuda::ORB> descriptor_;

    cv::Ptr<cv::AKAZE> akaze_;
    cv::Ptr<cv::BRISK> brisk_;


};

#endif //VO_FEATURES_CUDA_DETECTION_H
