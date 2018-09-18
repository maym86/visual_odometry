
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

void run(float offset){
    cv::Mat K = cv::Mat::eye(3,3,CV_64FC1);

    K.at<double>(0,0) = 718.856;
    K.at<double>(1,1) = 718.856;
    K.at<double>(0,2) = 607.1928;
    K.at<double>(1,2) = 185.2157;

    BundleAdjustment ba;

    VOFrame vo0;
    VOFrame vo1;
    VOFrame vo2;

    vo0.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    vo2.image = cv::imread("../src/sfm/test/test_data/000003.png");

    EXPECT_GT(vo0.image.rows, 0);
    EXPECT_GT(vo1.image.rows, 0);
    EXPECT_GT(vo2.image.rows, 0);

    double data[3] = {0,0,0};
    cv::Mat r45 = cv::Mat(3,1, CV_64F, data);
    vo0.pose_R = eulerAnglesToRotationMatrix(r45);

    LOG(INFO) << vo0.pose_R;
    vo0.pose_t = cv::Mat::zeros(3, 1, CV_64FC1);

    vo0.pose_t.at<double>(0,0) += offset;
    vo0.pose_t.at<double>(1,0) += offset;
    vo0.pose_t.at<double>(2,0) += offset;

    hconcat(vo0.pose_R, vo0.pose_t, vo0.pose);

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    feature_detector.detectFAST(&vo0);
    feature_tracker.trackPoints(&vo0, &vo1);

    updatePose(K, &vo0, &vo1);

    feature_detector.detectFAST(&vo1);
    feature_tracker.trackPoints(&vo1, &vo2);

    updatePose(K, &vo1, &vo2);

    ba.init(K , 3);

    ba.addKeyFrame(vo0);
    ba.addKeyFrame(vo1);
    ba.addKeyFrame(vo2);

    ba.draw(1);

    ba.drawViz();
    cv::waitKey(0);

    cv::Mat R, t;
    ba.slove(&R, &t);

    LOG(INFO) << vo2.pose_t;
    LOG(INFO) << t;

    double dist = cv::norm(vo2.pose_t - t);
    LOG(INFO) << dist;
    EXPECT_NEAR(dist, 0, 0.1);

    ba.drawViz();
    ba.draw(1);
    cv::waitKey(0);
}

TEST(BundleAdjustmentTest, Passes) {
    run(0);
    run(50);
}

