
#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"
#include "src/utils/draw.h"

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#include "src/features/cuda/feature_tracker.h"
#else
#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#endif

#include <glog/logging.h>

cv::Point2d pp(607.1928, 185.2157);
cv::Point2d focal(718.856, 718.856);

const float kMax3DDist = 200;



void run(VOFrame &vo0, VOFrame &vo1) {

    cv::Mat K = cv::Mat::eye(3,3,CV_64F);

    K.at<double>(0,0) = focal.x;
    K.at<double>(1,1) = focal.y;
    K.at<double>(0,2) = pp.x;
    K.at<double>(1,2) = pp.y;

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    feature_detector.detectFAST(&vo0);

    feature_tracker.trackPoints(&vo0, &vo1);

    vo1.E = cv::findEssentialMat(vo0.points, vo1.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo1.mask);
    recoverPose(vo1.E, vo0.points, vo1.points, vo1.local_R, vo1.local_t, focal.x, pp, vo1.mask);

    std::vector<cv::Point2f> p0, p1;
    for (int i = 0; i < vo0.points.size(); i ++){
        if(vo1.mask.at<bool>(i)){
            p0.push_back(vo0.points[i]);
            p1.push_back(vo1.points[i]);
        }
    }
    hconcat(vo1.local_R, vo1.local_t, vo1.local_P);

    LOG(INFO) << vo1.local_P;

    vo1.points_3d =  triangulate(p0, p1, K * cv::Mat::eye(3, 4, CV_64FC1), K * vo1.local_P);

    //filter
    for (int i = vo1.points_3d.size() - 1; i >= 0; --i) {
        if (vo1.mask.at<bool>(i) && vo1.points_3d[i].z > 0 &&
            cv::norm(vo1.points_3d[i] - cv::Point3d(0, 0, 0)) < kMax3DDist) {
            continue;
        }

        vo1.points_3d.erase(vo1.points_3d.begin() + i);
    }

    draw3D("method1", vo1.points_3d, 10);

    cv::waitKey(1);
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

