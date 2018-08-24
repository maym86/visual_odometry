

#ifndef VO_VO_FRAME_H
#define VO_VO_FRAME_H


struct VOFrame {

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

    cv::cuda::GpuMat gpu_image;
    std::vector<cv::Point2f> points;
    std::vector<int> tracked_index;

    std::vector<cv::Point3d> points_3d;

    VOFrame& operator =(const VOFrame& a) {
        E = a.E.clone();
        R = a.R.clone();
        t = a.t.clone();
        P = a.P.clone();
        mask = a.mask.clone();
        pose_R = a.pose_R.clone();
        pose_t = a.pose_t.clone();
        pose = a.pose.clone();
        gpu_image = a.gpu_image.clone();
        points = a.points;
        points_3d = a.points_3d;
        return *this;
    }
};


#endif //VO_VO_FRAME_H
