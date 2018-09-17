
#include <gtest/gtest.h>
#include "src/visual_odometry/vo_pose.h"

#include "src/utils/draw.h"

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#include "src/features/cuda/feature_tracker.h"
#else
#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#endif

#include <glog/logging.h>

void run(VOFrame &vo0, VOFrame &vo1) {

    cv::Mat K = cv::Mat::eye(3,3,CV_64F);

    K.at<double>(0,0) = 718.856;
    K.at<double>(1,1) = 718.856;
    K.at<double>(0,2) = 607.1928;
    K.at<double>(1,2) = 185.2157;

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    feature_detector.detectFAST(&vo0);

    feature_tracker.trackPoints(&vo0, &vo1);

    updatePose(K, &vo0, &vo1);

    //filter
    for (int i = vo1.points_3d.size() - 1; i >= 0; --i) {
        if (vo1.mask.at<bool>(i) && vo1.points_3d[i].z < 0 &&
            cv::norm(vo1.points_3d[i] - cv::Point3d(0, 0, 0)) < kMax3DDist) {
            continue;
        }

        vo1.points_3d.erase(vo1.points_3d.begin() + i);
    }

    draw3D("method1", vo1.points_3d, 10);

    cv::waitKey(0);
}

TEST(TriangulationTest, Passes) {

    VOFrame vo0;
    VOFrame vo1;

    vo0.image = cv::imread("../src/sfm/test/test_data/000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/000003.png");

    run(vo0,vo1);

}

TEST(TriangulationTestStereo, Passes) {


    VOFrame vo0;
    VOFrame vo1;

    vo0.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");

    run(vo0,vo1);
}

