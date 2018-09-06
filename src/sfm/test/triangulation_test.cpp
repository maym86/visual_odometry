
#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"

#include <glog/logging.h>

double P0[12] = {1.5334888857352591e+003, 0., 1.1192188110351562e+003, 0.,
                 0., 1.5334888857352591e+003, 9.1114062500000000e+002, 0.,
                 0., 0., 1., 0.};

double P1[12] = {1.5334888857352591e+003, 0., 1.1192188110351562e+003, 4.3953258083993223e+003,
                 0., 1.5334888857352591e+003, 9.1114062500000000e+002, 0.,
                 0., 0., 1., 0. };

cv::Mat K = (cv::Mat_<double>(3,3) <<   718.856, 0, 607.1928,
        0, 718.856, 185.2157,
        0, 0, 1);

TEST(TriangulationTest, Passes) {

    VOFrame vo0;
    VOFrame vo1;

    LOG(INFO) << K * cv::Mat::eye(3, 4, CV_64FC1);

    vo0.local_P = K * cv::Mat(3, 4, CV_64FC1, P0);
    vo1.local_P = K * cv::Mat(3, 4, CV_64FC1, P1);

    vo0.points.emplace_back(cv::Point2f(919,686));
    vo1.points.emplace_back(cv::Point2f(586,694));

    triangulateFrame(vo0, &vo1);

    LOG(INFO) << vo1.points_3d[0];
}

