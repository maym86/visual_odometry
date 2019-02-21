
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <src/utils/utils.h>

#include "src/sfm/bundle_adjustment.h"
#include "src/visual_odometry/vo_pose.h"

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#include "src/features/cuda/feature_tracker.h"
#else
#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#endif

#include "src/utils/draw.h"

void run(std::vector<VOFrame> &frames, float offset){
    cv::Mat K = cv::Mat::eye(3,3,CV_64FC1);

    K.at<double>(0,0) = 718.856;
    K.at<double>(1,1) = 718.856;
    K.at<double>(0,2) = 607.1928;
    K.at<double>(1,2) = 185.2157;

    BundleAdjustment ba;

    frames[0].pose_R = cv::Mat::eye(3,3, CV_64F);

    LOG(INFO) << frames[0].pose_R;
    frames[0].pose_t = cv::Mat::zeros(3, 1, CV_64FC1);

    frames[0].pose_t.at<double>(0,0) += offset;
    frames[0].pose_t.at<double>(1,0) += offset;
    frames[0].pose_t.at<double>(2,0) += offset;

    hconcat(frames[0].pose_R, frames[0].pose_t, frames[0].pose);

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    ba.init(K , frames.size());

    for(int i = 0; i < frames.size()-1; i++) {
        feature_detector.detectFAST(&frames[i]);
        feature_tracker.trackPoints(&frames[i], &frames[i + 1]);
        updatePose(K, &frames[i], &frames[i + 1]);
    }

    for(int i = 0; i < frames.size(); i++) {
        ba.addKeyFrame(frames[i]);
    }

    ba.draw(10);
    ba.drawViz();
    cv::waitKey(0);
    cv::Mat R, t;

    ba.slove(&R, &t);
    ba.drawViz();
    ba.draw(10);

    cv::waitKey(0);
}

TEST(BundleAdjustmentTest, Passes) {
    std::vector<VOFrame> frames(5);

    frames[0].image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");
    frames[1].image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    frames[2].image = cv::imread("../src/sfm/test/test_data/000001.png");
    frames[3].image = cv::imread("../src/sfm/test/test_data/000003.png");
    frames[4].image = cv::imread("../src/sfm/test/test_data/000007.png");

    run(frames, 0);
}


TEST(BundleAdjustmentTestStraight, Passes) {
    std::vector<VOFrame> frames(3);

    frames[0].image = cv::imread("../src/sfm/test/test_data/000000.png");
    frames[1].image = cv::imread("../src/sfm/test/test_data/000003.png");
    frames[2].image = cv::imread("../src/sfm/test/test_data/000007.png");

    run(frames, 0);
}


