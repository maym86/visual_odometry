#include "visual_odometry.h"

#include "src/sfm/triangulation.h"

#include <glog/logging.h>


VisualOdometry::VisualOdometry(float focal, const cv::Point2d &pp, size_t min_tracked_points) {
    tracking_ = false;
    focal_ = focal;
    pp_ = pp;

    min_tracked_points_ = min_tracked_points;
    last_keyframe_t_ = cv::Mat::zeros(3, 1, CV_64F); //TODO init elswhere so first point is added
    frame_buffer_ = boost::circular_buffer<VOFrame>(kFrameBufferCapacity);
    bundle_adjustment_.init(focal, pp, 10);

#if __has_include("opencv2/viz.hpp")
    window_ = cv::viz::Viz3d("Window");
    window_.setWindowSize(cv::Size(800,800));
    window_.setWindowPosition(cv::Point(150,150));
    window_.setBackgroundColor(); // black by default
#endif
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
            //This finds good correspondences (mask) using RANSAC - we already have ProjectionMat from vo0 to vo1
            cv::findEssentialMat(vo1.points, vo0.points, focal_, pp_, cv::RANSAC, 0.999, 1.0, vo1.mask);
            triangulateFrame(pp_, focal_, vo0, &vo1);
        }
        tracking_ = true;
    }

    feature_tracker_.trackPoints(&vo1, &vo2);

    if (vo2.points.size() < min_tracked_points_) {
        tracking_ = false;
    }

    vo2.E = cv::findEssentialMat(vo2.points, vo1.points, focal_, pp_, cv::RANSAC, 0.999, 1.0, vo2.mask);
    int res = recoverPose(vo2.E, vo2.points, vo1.points, vo2.local_R, vo2.local_t, focal_, pp_, vo2.mask);

    if (res > kMinPosePoints) {
        hconcat(vo2.local_R, vo2.local_t, vo2.local_P);
        triangulateFrame(pp_, focal_, vo1, &vo2);

        vo2.scale = getScale(vo1, vo2, kMinPosePoints, 200);
        LOG(INFO) << "Scale: " << vo2.scale;

        vo2.pose_R = vo2.local_R * vo1.pose_R;
        vo2.pose_t = vo1.pose_t + vo2.scale * (vo1.pose_R * vo2.local_t);
    } else {
        //Copy last pose
        LOG(INFO) << "RecoverPose, too few points";
        vo2.pose_R = vo1.pose_R.clone();
        vo2.pose_t = vo1.pose_t.clone();
    }
    hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);

    LOG(INFO) << vo2.local_P;
    /*if (cv::norm(last_keyframe_t_ - vo2.pose_t) > 3) {
        bundle_adjustment_.addKeyFrame(vo2);
        res = bundle_adjustment_.slove(&vo2.pose_R, &vo2.pose_t);

        if (res == 0) {
            hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);
        }
        last_keyframe_t_ = vo2.pose_t;
    }*/

    //Kalman Filter
    //kf_.setMeasurements(vo2.pose_R, vo2.pose_t);
    //cv::Mat k_R, k_t;
    //kf_.updateKalmanFilter(&k_R, &k_t);

    //hconcat(k_R, k_t, *pose_kalman);

    (*pose) = vo2.pose;
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

    cv::Mat drawXY(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(drawXY, cv::Point(drawXY.cols / 2, 0), cv::Point(drawXY.cols / 2, drawXY.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXY, cv::Point(0, drawXY.rows / 2), cv::Point(drawXY.cols, drawXY.rows / 2), cv::Scalar(0, 0, 255));

    if (frame_buffer_.full()) {
        VOFrame &vo2 = frame_buffer_[frame_buffer_.size() - 1];

#if __has_include("opencv2/viz.hpp")
        if(!vo2.points_3d.empty()) {
            cv::viz::WCloud cloud_widget(vo2.points_3d, cv::viz::Color::green());
            window_.showWidget("point_cloud", cloud_widget);
            window_.spinOnce();
        }
#endif

        for (int j = 0; j < vo2.points_3d.size(); j++) {

            cv::Point2d draw_pos = cv::Point2d(vo2.points_3d[j].x * vo2.scale + drawXY.cols / 2,
                                               vo2.points_3d[j].y * vo2.scale + drawXY.rows / 2);

            cv::circle(drawXY, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
        }
    }
    return drawXY;
}