
#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"
#include "src/utils/draw.h"

#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"

#include <glog/logging.h>


cv::Point2d pp(607.1928, 185.2157);
cv::Point2d focal(718.856, 718.856);

TEST(TriangulationTest, Passes) {

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    VOFrame vo0;
    VOFrame vo1;

    vo0.image = cv::imread("../src/sfm/test/test_data/000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/000001.png");

    feature_detector.detectFAST(&vo0);
    feature_tracker.trackPoints(&vo0, &vo1);

    vo1.E = cv::findEssentialMat(vo1.points, vo0.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo1.mask);
    recoverPose(vo1.E, vo1.points, vo0.points, vo1.local_R, vo1.local_t, focal.x, pp, vo1.mask);

    hconcat(vo1.local_R, vo1.local_t, vo1.local_P);

    LOG(INFO) << vo1.local_P;

    triangulateFrame(pp, focal, vo0, &vo1);
    draw3D("method1", vo1, 10);

    //Method 2

    cv::Mat K = cv::Mat::eye(3,3, CV_64FC1);

    K.at<double>(0,0) = focal.x;
    K.at<double>(1,1) = focal.y;
    K.at<double>(0,2) = pp.x;
    K.at<double>(1,2) = pp.y;

    cv::Mat points3d;
    recoverPose(vo1.E, vo1.points, vo0.points, K, vo1.local_R, vo1.local_t, 500, vo1.mask, points3d);
    vo1.points_3d = points3dToVec(points3d);

    LOG(INFO) << vo1.local_P;

    draw3D("method2", vo1, 100);
    cv::waitKey(0);

}

TEST(TriangulationTestStereo, Passes) {
    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    VOFrame vo0;
    VOFrame vo1;

    vo0.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");

    feature_detector.detectFAST(&vo0);

    feature_tracker.trackPoints(&vo0, &vo1);

    vo1.E = cv::findEssentialMat(vo1.points, vo0.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo1.mask);
    int res = recoverPose(vo1.E, vo1.points, vo0.points, vo1.local_R, vo1.local_t, focal.x, pp, vo1.mask);

    hconcat(vo1.local_R, vo1.local_t, vo1.local_P);

    LOG(INFO) << vo1.local_P;

    triangulateFrame(pp, focal, vo0, &vo1);
    draw3D("method1", vo1, 10);

    //Method 2

    cv::Mat K = cv::Mat::eye(3,3, CV_64FC1);

    K.at<double>(0,0) = focal.x;
    K.at<double>(1,1) = focal.y;
    K.at<double>(0,2) = pp.x;
    K.at<double>(1,2) = pp.y;

    cv::Mat points3d;
    recoverPose(vo1.E, vo1.points, vo0.points, K, vo1.local_R, vo1.local_t, 500, vo1.mask, points3d);
    vo1.points_3d = points3dToVec(points3d);

    LOG(INFO) << vo1.local_P;

    draw3D("method2", vo1, 100);
    cv::waitKey(0);
}

