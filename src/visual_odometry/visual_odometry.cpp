#include "visual_odometry.h"

#include <glog/logging.h>

#include "src/sfm/triangulation.h"
#include "src/utils/utils.h"
#include "vo_pose.h"

VisualOdometry::VisualOdometry(const cv::Mat &K, size_t min_tracked_points, bool ba) {
    tracking_ = false;

    min_tracked_points_ = min_tracked_points;
    frame_buffer_ = boost::circular_buffer<VOFrame>(kFrameBufferCapacity);
    bundle_adjustment_.init(K, kBundleWindow);
    ba_ = ba;

    K_ = K.clone();
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
        tracking_ = true;
    }

    if (feature_tracker_.trackPoints(&vo1, &vo2) < 0) { //TODO set correctly
        frame_buffer_.erase(frame_buffer_.end() - 1);
        return;
    }


    if (vo2.points.size() < min_tracked_points_) {
        tracking_ = false;
    }

    int res = updatePose(K_, &vo1, &vo2);

    if(res == 0 && ba_) { //TODO this doesnt work well yet
        bundle_adjustment_.addKeyFrame(vo2);
        res = bundle_adjustment_.slove(&vo2.pose_R, &vo2.pose_t);
        bundle_adjustment_.draw(1);

        if (res == 0) {
            hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);
        }
    }
    //Kalman Filter
    //kf_.setMeasurements(vo2.pose_R, vo2.pose_t);
    //cv::Mat k_R, k_t;
    //kf_.updateKalmanFilter(&k_R, &k_t);

    //hconcat(k_R, k_t, *pose_kalman);

    (*pose) = vo2.pose.clone();
}

cv::Mat VisualOdometry::drawMatches(const cv::Mat &image) {

    cv::Mat output = image.clone();

    if (frame_buffer_.full()) {
        VOFrame &vo1 = frame_buffer_[frame_buffer_.size() - 2];
        VOFrame &vo2 = frame_buffer_[frame_buffer_.size() - 1];

        for (int i = 0; i < vo1.points.size(); i++) {
            if (vo2.mask.at<bool>(i)) {
                cv::line(output, vo1.points[i], vo2.points[i], color_, 2);
            }
        }
    }
    return output;
}


cv::Mat VisualOdometry::draw3D() {

    cv::Mat drawXY(500, 1000, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(drawXY, cv::Point(drawXY.cols / 2, 0), cv::Point(drawXY.cols / 2, drawXY.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXY, cv::Point(0, drawXY.rows / 2), cv::Point(drawXY.cols, drawXY.rows / 2), cv::Scalar(0, 0, 255));

    if (frame_buffer_.full()) {
        VOFrame &vo2 = frame_buffer_[frame_buffer_.size() - 1];

        std::vector<cv::Point3d> inliers;
        for (int j = 0; j < vo2.points_3d.size(); j++) {

            if(vo2.mask.at<bool>(j) && vo2.points_3d[j].z > 0 && cv::norm(vo2.points_3d[j]) < kMax3DDist) {
                cv::Point2d draw_pos = cv::Point2d(vo2.points_3d[j].x * vo2.scale * 20 + drawXY.cols / 2,
                                                   vo2.points_3d[j].y * vo2.scale * 20 + drawXY.rows / 2);

                inliers.push_back(vo2.points_3d[j]);
                cv::circle(drawXY, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
            }
        }

    }
    return drawXY;
}


