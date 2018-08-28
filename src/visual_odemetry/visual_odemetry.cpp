#include "visual_odemetry.h"

#include "src/sfm/triangulation.h"

#include <glog/logging.h>

VisualOdemetry::VisualOdemetry(double focal, const cv::Point2d &pp) {
    tracking_ = false;
    focal_ = focal;
    pp_ = pp;

    frame_buffer_ = boost::circular_buffer<VOFrame>(kFrameBufferCapacity);
}

void VisualOdemetry::addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman){

    VOFrame frame;
    frame.setImage(image);
    frame_buffer_.push_back(std::move(frame));
    if (!frame_buffer_.full()) {
        return;
    }

    VOFrame &vo1 = frame_buffer_[1];
    VOFrame &vo2 = frame_buffer_[2];

    bool new_keypoints =  false;
    color_ = cv::Scalar(0,0,255);
    if (!tracking_) {
        color_ = cv::Scalar(255,0,0);
        feature_detector_.detect(&vo1);

        VOFrame &vo0 = frame_buffer_[0];
        if(!vo0.image.empty() && !vo1.E.empty()){ //Backtrack for new points for scale calculation later
            feature_tracker_.trackPoints(&vo1, &vo0);
            //This finds good correspondences (mask) using RANSAC - we already have R|t from vo0 to vo1
            cv::findEssentialMat( vo1.points, vo0.points, focal_, pp_, cv::RANSAC, 0.999, 1.0, vo1.mask);
            triangulate(&vo0, &vo1);
        }
        tracking_ = true;
        new_keypoints = true;
    }

    feature_tracker_.trackPoints(&vo1, &vo2);

    if (vo2.points.size() < kMinTrackedPoints) {
        tracking_ = false;
    }

    vo2.E = cv::findEssentialMat( vo2.points, vo1.points,  focal_, pp_, cv::RANSAC, 0.999, 1.0, vo2.mask);
    int res = recoverPose(vo2.E, vo2.points, vo1.points, vo2.R, vo2.t, focal_, pp_, vo2.mask);

    if(res > kMinPosePoints) {
        hconcat(vo2.R, vo2.t, vo2.P);
        triangulate(&vo1, &vo2);

        double scale = getScale(vo1, vo2, kMinPosePoints, 200);
        LOG(INFO) << new_keypoints << " " << scale;

        vo2.pose_R = vo2.R * vo1.pose_R;
        vo2.pose_t = vo1.pose_t + scale * (vo1.pose_R * vo2.t);
    } else {
        //Copy last pose
        vo2.pose_R = vo1.pose_R.clone();
        vo2.pose_t = vo1.pose_t.clone();
    }
    hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);

    //Kalman Filter
    kf_.setMeasurements(vo2.pose_R, vo2.pose_t);
    cv::Mat k_R, k_t;
    kf_.updateKalmanFilter(&k_R, &k_t);

    hconcat(k_R, k_t, *pose_kalman);

    (*pose) = vo2.pose;
    //TODO keep sliding window and use bundle adjustment to correct pos of last frame
}

cv::Mat VisualOdemetry::drawMatches(const cv::Mat &image){

    cv::Mat output = image.clone();

    if(frame_buffer_.full()) {
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

