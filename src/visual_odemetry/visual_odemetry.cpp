#include "visual_odemetry.h"
#include <glog/logging.h>

VisualOdemetry::VisualOdemetry(double focal, const cv::Point2d &pp) {
    pose_t_ = cv::Mat::zeros(3, 1, CV_64FC1);
    pose_R_ = cv::Mat::eye(3, 3, CV_64FC1);
    tracking_ = false;
    focal_ = focal;
    pp_ = pp;
}

void VisualOdemetry::addImage(const cv::Mat &image, cv::Mat *pose){
    //Store previous frame data
    points_previous_ = points_;
    prev_gpu_image_ = gpu_image_.clone();
    prev_pose_ = pose_;

    //Get new GPU image
    cv::Mat image_grey;
    cv::cvtColor(image, image_grey, CV_BGR2GRAY);
    gpu_image_.upload(image_grey);

    if (prev_gpu_image_.empty()) {
        prev_gpu_image_ = gpu_image_.clone();
        return;
    }

    color_ = cv::Scalar(255,0,0);
    if (!tracking_) {

        color_ = cv::Scalar(0,0,255);
        points_previous_ = feature_detector_.detect(prev_gpu_image_);
        tracking_ = true;
    }

    points_ = feature_tracker_.trackPoints(prev_gpu_image_, gpu_image_, &points_previous_);

    if (points_.size() < kMinTrackedPoints) {
        tracking_ = false;
    }

    cv::Mat E, R, t;

    E = cv::findEssentialMat( points_, points_previous_,  focal_, pp_, cv::RANSAC, 0.999, 1.0, mask_);
    int res = recoverPose(E, points_, points_previous_,   R, t, focal_, pp_, mask_);

    if(res > 10) {
        pose_R_ = R * pose_R_;
        pose_t_ += kScale * (pose_R_ * t);
        hconcat(pose_R_, pose_t_, pose_);
    }

    *pose = pose_;

    LOG(INFO) << "\n" << pose_;


}

cv::Mat VisualOdemetry::drawMatches(const cv::Mat &image){
    cv::Mat output = image.clone();
    for (int i = 0; i < points_previous_.size(); i++) {
        if (mask_.at<bool>(i)) {
            cv::line(output, points_previous_[i], points_[i], color_, 2);
        }
    }

    return output;
}