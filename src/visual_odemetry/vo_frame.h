

#ifndef VO_VO_FRAME_H
#define VO_VO_FRAME_H

#include <cv.hpp>
#include <opencv2/core/cuda.hpp>

class VOFrame {

public:
    //Local transform between frames
    cv::Mat E; // Essential matrix
    cv::Mat R; // Rotation
    cv::Mat t; // Translation
    cv::Mat P; // Projection matrix R|t
    cv::Mat mask;

    //Global pose
    cv::Mat pose_R = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat pose_t = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat pose; //global

    cv::Mat image;
    cv::cuda::GpuMat gpu_image;
    std::vector<cv::Point2f> points;
    std::vector<int> tracked_index;

    std::vector<cv::Point3d> points_3d;

    void setImage(cv::Mat in){
        image = in;
#ifdef HAVE_CUDA
        cv::Mat image_grey;
        cv::cvtColor(image, image_grey, CV_BGR2GRAY);
        gpu_image.upload(image_grey);
#endif
    };

    VOFrame& operator =(const VOFrame& other) {
        E = other.E.clone();
        R = other.R.clone();
        t = other.t.clone();
        P = other.P.clone();
        mask = other.mask.clone();
        pose_R = other.pose_R.clone();
        pose_t = other.pose_t.clone();
        pose = other.pose.clone();
        image = other.image.clone();
        points = other.points;
        points_3d = other.points_3d;
#ifdef HAVE_CUDA
        gpu_image = a.gpu_image.clone();
#endif
        return *this;
    }
};


#endif //VO_VO_FRAME_H
