
#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"

#include <glog/logging.h>


double R[12] = {0.9999899726688515, -0.0002398295194755593, -0.004471805401671894, -10.001475811533083541,
        0.0002404186852745609, 0.9999999624908545, 0.0001312141258476285, -10.02857142671942455,
        0.004471773764917525, -0.0001322879156956044, 0.9999899928195798, 10.9995906639997874};
/*
double P1[12] = {1, 0., 0, 4.3953258083993223e+002,
                 0., 1, 0, 0.,
                 0., 0., 1., 0. };

*/
cv::Point2d pp(607.1928, 185.2157);
double focal = 718.856;

TEST(TriangulationTest, Passes) {

    VOFrame vo0;
    VOFrame vo1;

    vo0.local_P = cv::Mat::eye(3, 4, CV_64FC1);
    vo1.local_P = cv::Mat(3, 4, CV_64FC1, P1);

    vo0.points.emplace_back(cv::Point2f(919,686));
    vo0.points.emplace_back(cv::Point2f(919+100,686));
    vo1.points.emplace_back(cv::Point2f(586,694));
    vo1.points.emplace_back(cv::Point2f(586+100,694));

    triangulateFrame(pp, focal, vo0, &vo1);

    LOG(INFO) << vo1.points_3d[0];
    LOG(INFO) << vo1.points_3d[1];

    cv::Mat drawXY(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(drawXY, cv::Point(drawXY.cols / 2, 0), cv::Point(drawXY.cols / 2, drawXY.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXY, cv::Point(0, drawXY.rows / 2), cv::Point(drawXY.cols, drawXY.rows / 2), cv::Scalar(0, 0, 255));


    cv::Point2d draw_pos = cv::Point2d(P1[3] + drawXY.cols / 2, P1[7]  + drawXY.rows / 2);

    cv::circle(drawXY, draw_pos, 3, cv::Scalar(255, 0, 0), 3);



    for (int j = 0; j < vo1.points_3d.size(); j++) {
        cv::Point2d draw_pos = cv::Point2d(vo1.points_3d[j].x + drawXY.cols / 2,
                                           vo1.points_3d[j].y + drawXY.rows / 2);

        cv::circle(drawXY, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow("p3d", drawXY);
    cv::waitKey(0);
}

