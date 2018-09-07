

#ifndef VO_VO_FRAME_H
#define VO_VO_FRAME_H

#include <cv.hpp>
#include <opencv2/core/cuda.hpp>

#if __has_include("opencv2/cudafeatures2d.hpp")
const bool kHasCUDA = true;
#else
const bool kHasCUDA = false;
#endif

class VOFrame {
public:
    //Local transform between frames
    cv::Mat E; // Essential matrix
    cv::Mat local_R; // Rotation
    cv::Mat local_t; // Translation
    cv::Mat mask;

    //Global pose
    cv::Mat pose_R = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat pose_t = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat local_P; // Projection matrix K[R|t]
    cv::Mat pose; //global

    float scale;

    cv::Mat image;
    cv::cuda::GpuMat gpu_image;
    std::vector<cv::Point2f> points;
    std::vector<int> tracked_index;

    std::vector<cv::Point3d> points_3d;

    void setImage(cv::Mat image_in){
        image = image_in;
        if(kHasCUDA) {
            cv::Mat image_grey;
            cv::cvtColor(image, image_grey, CV_BGR2GRAY);
            gpu_image.upload(image_grey);
        }
    };

    VOFrame& operator =(const VOFrame& a) {
        E = a.E.clone();
        local_R = a.local_R.clone();
        local_t = a.local_t.clone();
        local_P = a.local_P.clone();
        mask = a.mask.clone();
        pose_R = a.pose_R.clone();
        pose_t = a.pose_t.clone();
        pose = a.pose.clone();
        scale = scale;
        image = image.clone();
        gpu_image = a.gpu_image.clone();
        points = a.points;
        points_3d = a.points_3d;
        tracked_index = a.tracked_index;
        return *this;
    }
};


#endif //VO_VO_FRAME_H
