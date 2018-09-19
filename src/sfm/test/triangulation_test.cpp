#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"
#include "src/utils/draw.h"

#include "src/features/feature_detector.h"
#include "src/features/feature_tracker.h"
#include "src/utils/utils.h"
#include "src/visual_odometry/vo_pose.h"

#include <glog/logging.h>

#include <opencv2/sfm/triangulation.hpp>

cv::Point2d pp(607.1928, 185.2157);
cv::Point2d focal(718.856, 718.856);

const int kDrawScale = 5;

void filter(const VOFrame &vo0, VOFrame *vo1){
    //filter

    cv::Mat R = vo0.pose.colRange(cv::Range(0,3));
    cv::Mat t = vo0.pose.col(3);

    std::vector<cv::Point3d> origin;
    for (int i = vo1->points_3d.size() - 1; i >= 0; --i) {

        cv::Mat p(vo1->points_3d[i]);
        p = (R.t() * p) - t;

        if (vo1->mask.at<bool>(i) && cv::norm(p) < 200) {// && p.at<double>(2) < t.at<double>(2)) { //TODO this is wrong
            continue;
        }
        vo1->points_3d.erase(vo1->points_3d.begin() + i);
    }

    draw3D("origin", origin, kDrawScale, vo0.pose, vo1->pose);
}

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

    updatePose(K, &vo0, &vo1);

    cv::Mat neg =  cv::Mat::eye(3,3,CV_64F);

    std::vector<cv::Point2f> p0, p1;
    for (int i = 0; i < vo0.points.size(); i ++){
        if(vo1.mask.at<bool>(i)){
            p0.push_back(vo0.points[i]);
            p1.push_back(vo1.points[i]);
        }
    }

    LOG(INFO) << vo0.pose << vo1.pose;

    vo1.points_3d =  triangulate(p0, p1, K * vo0.pose, K * vo1.pose);

    filter(vo0, &vo1);

    draw3D("method1", vo1.points_3d, kDrawScale, vo0.pose, vo1.pose);

    //Method 3
    cv::Mat points1Mat = (cv::Mat_<double>(2,1) << 1, 1);
    cv::Mat points2Mat = (cv::Mat_<double>(2,1) << 1, 1);

    for (int i=0; i < vo0.points.size(); i++) {
        cv::Mat matPoint1 = (cv::Mat_<double>(2,1) << p0[i].x, p0[i].y);
        cv::Mat matPoint2 = (cv::Mat_<double>(2,1) << p1[i].x, p1[i].y);
        cv::hconcat(points1Mat, matPoint1, points1Mat);
        cv::hconcat(points2Mat, matPoint2, points2Mat);
    }

    std::vector<cv::Mat> sfm_points_2d;
    sfm_points_2d.push_back(points1Mat);
    sfm_points_2d.push_back(points2Mat);

    std::vector<cv::Mat> sfm_proj_mats;
    sfm_proj_mats.push_back(K * vo0.pose);
    sfm_proj_mats.push_back(K * vo1.pose);

    cv::Mat points_3d_mat;
    cv::sfm::triangulatePoints(sfm_points_2d, sfm_proj_mats, points_3d_mat);
    vo1.points_3d = points3DToVec(points_3d_mat);
    filter(vo0, &vo1);

    draw3D("method2", vo1.points_3d, kDrawScale, vo0.pose, vo1.pose);

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


TEST(TriangulationTestStereoOffset, Passes) {
    VOFrame vo0;
    VOFrame vo1;

    for (int i = 0; i < 64; i++) {
        double data[3] = {0, static_cast<double>(i)/10.0, 0};
        cv::Mat euler = cv::Mat(3, 1, CV_64F, data);
        vo0.pose_R = eulerAnglesToRotationMatrix(euler);

        LOG(INFO) << vo0.pose_R;
        vo0.pose_t = cv::Mat::zeros(3, 1, CV_64FC1);

        vo0.pose_t.at<double>(0, 0) += 0;
        vo0.pose_t.at<double>(1, 0) += 0;
        vo0.pose_t.at<double>(2, 0) += 0;

        hconcat(vo0.pose_R, vo0.pose_t, vo0.pose);
        LOG(INFO) << vo0.pose_t;

        vo0.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
        vo1.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");

        run(vo0, vo1);
    }
}
