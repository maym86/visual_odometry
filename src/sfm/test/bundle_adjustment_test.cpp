
#include <gtest/gtest.h>

#include "src/sfm/bundle_adjustment.h"

#include <glog/logging.h>

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#include "src/features/cuda/feature_tracker.h"
#else
#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#endif

#include "src/utils/draw.h"

TEST(BundleAdjustmentTest, Passes) {

    cv::Point2d pp(607.1928, 185.2157);
    cv::Point2d focal(718.856, 718.856);

    BundleAdjustment ba;

    VOFrame vo0;
    VOFrame vo1;
    VOFrame vo2;

    vo0.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");
    vo2.image = cv::imread("../src/sfm/test/test_data/000003.png");

    EXPECT_GT(vo0.image.rows, 0);
    EXPECT_GT(vo1.image.rows, 0);
    EXPECT_GT(vo2.image.rows, 0);

    vo0.pose_R = cv::Mat::eye(3, 3, CV_64FC1);
    vo0.pose_t = cv::Mat::zeros(3, 1, CV_64FC1);

    hconcat(vo0.pose_R, vo0.pose_t, vo0.pose);

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    feature_detector.detectFAST(&vo0);
    feature_tracker.trackPoints(&vo0, &vo1);

    vo1.E = cv::findEssentialMat(vo0.points, vo1.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo1.mask);
    recoverPose(vo1.E, vo0.points, vo1.points, vo1.local_R, vo1.local_t, focal.x, pp, vo1.mask);

    vo1.pose_R = vo1.local_R.clone();
    vo1.pose_t = vo1.local_t.clone();

    hconcat(vo1.pose_R, vo1.pose_t, vo1.pose);

    feature_detector.detectFAST(&vo0);
    feature_tracker.trackPoints(&vo0, &vo2);

    vo2.E = cv::findEssentialMat(vo0.points, vo2.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo2.mask);
    recoverPose(vo2.E, vo0.points, vo2.points, vo2.local_R, vo2.local_t, focal.x, pp, vo2.mask);

    vo2.pose_R = vo2.local_R.clone();
    vo2.pose_t = vo2.local_t.clone();
    hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);

    ba.init(cv::Point2f(718.856,718.856), cv::Point2f(607.193, 185.216) , 3);

    ba.addKeyFrame(vo0);
    ba.addKeyFrame(vo1);
    ba.addKeyFrame(vo2);

    ba.draw(15);

    cv::waitKey(0);
    cv::Mat R, t;
    ba.slove(&R, &t);

    LOG(INFO) << vo2.pose_t;
    LOG(INFO) << t;

    double dist = cv::norm(vo2.pose_t - t);
    LOG(INFO) << dist;
    EXPECT_NEAR(dist, 0, 0.1);

    ba.draw(15);
    cv::waitKey(0);
}

TEST(BundleAdjustmentTestOffset, Passes) {

    cv::Point2d pp(607.1928, 185.2157);
    cv::Point2d focal(718.856, 718.856);

    BundleAdjustment ba;

    VOFrame vo0;
    VOFrame vo1;
    VOFrame vo2;

    vo0.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");
    vo2.image = cv::imread("../src/sfm/test/test_data/000003.png");

    EXPECT_GT(vo0.image.rows, 0);
    EXPECT_GT(vo1.image.rows, 0);
    EXPECT_GT(vo2.image.rows, 0);

    vo0.pose_R = cv::Mat::eye(3, 3, CV_64FC1);
    vo0.pose_t = cv::Mat::zeros(3, 1, CV_64FC1);
    vo0.pose_t.at<double>(0,0) += 10;
    vo0.pose_t.at<double>(1,0) += 10;
    vo0.pose_t.at<double>(2,0) += 10;

    hconcat(vo0.pose_R, vo0.pose_t, vo0.pose);

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    feature_detector.detectFAST(&vo0);
    feature_tracker.trackPoints(&vo0, &vo1);

    vo1.E = cv::findEssentialMat(vo0.points, vo1.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo1.mask);
    recoverPose(vo1.E, vo0.points, vo1.points, vo1.local_R, vo1.local_t, focal.x, pp, vo1.mask);

    vo1.pose_R = vo1.local_R.clone();
    vo1.pose_t = vo1.local_t.clone();
    vo1.pose_t.at<double>(0,0) += 10;
    vo1.pose_t.at<double>(1,0) += 10;
    vo1.pose_t.at<double>(2,0) += 10;

    hconcat(vo1.pose_R, vo1.pose_t, vo1.pose);

    feature_detector.detectFAST(&vo0);
    feature_tracker.trackPoints(&vo0, &vo2);

    vo2.E = cv::findEssentialMat(vo0.points, vo2.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo2.mask);
    recoverPose(vo2.E, vo0.points, vo2.points, vo2.local_R, vo2.local_t, focal.x, pp, vo2.mask);

    vo2.pose_R = vo2.local_R.clone();
    vo2.pose_t = vo2.local_t.clone();
    vo2.pose_t.at<double>(0,0) += 10;
    vo2.pose_t.at<double>(1,0) += 10;
    vo2.pose_t.at<double>(2,0) += 10;


    hconcat(vo2.pose_R, vo2.pose_t, vo2.pose);

    ba.init(cv::Point2f(718.856,718.856), cv::Point2f(607.193, 185.216) , 3);

    ba.addKeyFrame(vo0);
    ba.addKeyFrame(vo1);
    ba.addKeyFrame(vo2);

    ba.draw(15);

    cv::waitKey(0);
    cv::Mat R, t;
    ba.slove(&R, &t);

    LOG(INFO) << vo2.pose_t;
    LOG(INFO) << t;

    double dist = cv::norm(vo2.pose_t - t);
    LOG(INFO) << dist;
    EXPECT_NEAR(dist, 0, 0.1);

    ba.draw(15);
    cv::waitKey(0);
}
