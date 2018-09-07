
#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"
#include "src/utils/draw.h"

#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"

#include <glog/logging.h>

#include <opencv2/sfm/triangulation.hpp>

cv::Point2d pp(607.1928, 185.2157);
cv::Point2d focal(718.856, 718.856);


void run(VOFrame &vo0, VOFrame &vo1) {

    FeatureDetector feature_detector;
    FeatureTracker feature_tracker;

    cv::Mat K = cv::Mat::eye(3,3, CV_64FC1);

    K.at<double>(0,0) = focal.x;
    K.at<double>(1,1) = focal.y;
    K.at<double>(0,2) = pp.x;
    K.at<double>(1,2) = pp.y;


    feature_detector.detectFAST(&vo0);

    feature_tracker.trackPoints(&vo0, &vo1);

    vo1.E = cv::findEssentialMat(vo1.points, vo0.points, focal.x, pp, cv::RANSAC, 0.999, 1.0, vo1.mask);
    recoverPose(vo1.E, vo1.points, vo0.points, vo1.local_R, vo1.local_t, focal.x, pp, vo1.mask);

    std::vector<cv::Point2f> p0, p1;
    for (int i = 0; i < vo0.points.size(); i ++){
        if(vo1.mask.at<bool>(i)){
            p0.push_back(vo0.points[i]);
            p1.push_back(vo1.points[i]);
        }
    }
    hconcat(vo1.local_R, vo1.local_t, vo1.local_P);

    LOG(INFO) << vo1.local_P;

    vo1.points_3d =  triangulate(pp, focal, p0, p1, cv::Mat::eye(3, 4, CV_64FC1), vo1.local_P);
    draw3D("method1", vo1, 1);

    //Method 2
    cv::Mat points3d;
    recoverPose(vo1.E, vo1.points, vo0.points, K, vo1.local_R, vo1.local_t, 500, vo1.mask, points3d);
    vo1.points_3d = points3dToVec(points3d);

    draw3D("method2", vo1, 100);

    //Method 3
    cv::Mat points1Mat = (cv::Mat_<double>(2,1) << 1, 1);
    cv::Mat points2Mat = (cv::Mat_<double>(2,1) << 1, 1);

    for (int i=0; i < vo0.points.size(); i++) {
        cv::Mat matPoint1 = (cv::Mat_<double>(2,1) << p0[i].x, p0[i].y);
        cv::Mat matPoint2 = (cv::Mat_<double>(2,1) << p1[i].x, p1[i].y);
        cv::hconcat(points1Mat, matPoint1, points1Mat);
        cv::hconcat(points2Mat, matPoint2, points2Mat);
    }

    std::vector<cv::Mat> sfmPoints2d;
    sfmPoints2d.push_back(points1Mat);
    sfmPoints2d.push_back(points2Mat);

    std::vector<cv::Mat> sfmProjMats;
    sfmProjMats.push_back(K *  cv::Mat::eye(3, 4, CV_64FC1));
    sfmProjMats.push_back(K * vo1.local_P);

    cv::sfm::triangulatePoints(sfmPoints2d, sfmProjMats, points3d);
    vo1.points_3d = points3dToVec(points3d);

    draw3D("method3", vo1, 1);

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

