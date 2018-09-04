#include "visual_odometry.h"

#include "src/sfm/triangulation.h"

#include <glog/logging.h>

VisualOdometry::VisualOdometry(double focal, const cv::Point2d &pp, size_t min_tracked_points) {
    tracking_ = false;
    focal_ = focal;
    pp_ = pp;
    min_tracked_points_ = min_tracked_points;
    last_keyframe_t_ = cv::Mat::zeros(3, 1, CV_64F); //TODO init elswhere so first point is added
    frame_buffer_ = boost::circular_buffer<VOFrame>(kFrameBufferCapacity);
    bundle_adjustment_.init(10, pp);
}

void VisualOdometry::addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman) {

    VOFrame frame;
    frame.setImage(image);
    frame_buffer_.push_back(std::move(frame));
    if (!frame_buffer_.full()) {
        return;
    }

    VOFrame &vo1 = frame_buffer_[frame_buffer_.size() - 2];
    VOFrame &vo2 = frame_buffer_[frame_buffer_.size() - 1];

    color_ = cv::Scalar(0, 0, 255);
    if (!tracking_) {
        color_ = cv::Scalar(255, 0, 0);
        feature_detector_.detectFAST(&vo1);

        VOFrame &vo0 = frame_buffer_[frame_buffer_.size() - 3];
        if (!vo0.image.empty() && !vo1.E.empty()) { //Backtrack for new points for scale calculation later
            feature_tracker_.trackPoints(&vo1, &vo0);
            //This finds good correspondences (mask) using RANSAC - we already have R|t from vo0 to vo1
            cv::findEssentialMat(vo1.points, vo0.points, focal_, pp_, cv::RANSAC, 0.999, 1.0, vo1.mask);
            triangulateFrame(&vo0, &vo1);
        }
        tracking_ = true;
    }

    feature_tracker_.trackPoints(&vo1, &vo2);

    if (vo2.points.size() < min_tracked_points_) {
        tracking_ = false;
    }

    vo2.E = cv::findEssentialMat(vo2.points, vo1.points, focal_, pp_, cv::RANSAC, 0.999, 1.0, vo2.mask);
    int res = recoverPose(vo2.E, vo2.points, vo1.points, vo2.R, vo2.t, focal_, pp_, vo2.mask);

    if (res > kMinPosePoints) {
        hconcat(vo2.R, vo2.t, vo2.P);
        triangulateFrame(&vo1, &vo2);

        vo2.scale = getScale(vo1, vo2, kMinPosePoints, 200);
        LOG(INFO) << "Scale: " << vo2.scale;

        vo2.pose_R = vo2.R * vo1.pose_R;
        vo2.pose_t = vo1.pose_t + vo2.scale * (vo1.pose_R * vo2.t);
    } else {
        //Copy last pose
        LOG(INFO) << "RecoverPose, too few points";
        vo2.pose_R = vo1.pose_R.clone();
        vo2.pose_t = vo1.pose_t.clone();
    }
    hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);

    //Kalman Filter
    kf_.setMeasurements(vo2.pose_R, vo2.pose_t);
    cv::Mat k_R, k_t;
    kf_.updateKalmanFilter(&k_R, &k_t);

    hconcat(k_R, k_t, *pose_kalman);

    if (cv::norm(last_keyframe_t_ - vo2.pose_t) > 3) {
        bundle_adjustment_.addKeyFrame(vo2, focal_);
        int res = bundle_adjustment_.slove(&vo2.pose_R, &vo2.pose_t);

        if (res == 0) {
            hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);
        }
        last_keyframe_t_ = vo2.pose_t;
    }

    (*pose) = vo2.pose;
}

cv::Mat VisualOdometry::drawMatches(const cv::Mat &image) {

    cv::Mat output = image.clone();

    if (frame_buffer_.full()) {
        VOFrame &vo1 = frame_buffer_[1];
        VOFrame &vo2 = frame_buffer_[2];

        for (int i = 0; i < vo1.points.size(); i++) {
            if (vo2.mask.at<bool>(i)) {
                cv::line(output, vo1.points[i], vo2.points[i], color_, 2);
            }
        }
    }
    return output;
}

