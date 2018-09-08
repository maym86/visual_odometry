

#include "draw.h"


void draw3D(const std::string &name, VOFrame &frame, float scale){
    cv::Mat drawXY(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat drawXZ(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(drawXY, cv::Point(drawXY.cols / 2, 0), cv::Point(drawXY.cols / 2, drawXY.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXY, cv::Point(0, drawXY.rows / 2), cv::Point(drawXY.cols, drawXY.rows / 2), cv::Scalar(0, 0, 255));

    cv::line(drawXZ, cv::Point(drawXZ.cols / 2, 0), cv::Point(drawXZ.cols / 2, drawXZ.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXZ, cv::Point(0, drawXZ.rows / 2), cv::Point(drawXZ.cols, drawXZ.rows / 2), cv::Scalar(0, 0, 255));

    for (int j = 0; j < frame.points_3d.size(); j++) {
        cv::Point2d draw_pos = cv::Point2d(frame.points_3d[j].x * scale + drawXY.cols / 2,
                               frame.points_3d[j].y * scale + drawXY.rows / 2);

        cv::circle(drawXY, draw_pos, 1, cv::Scalar(0, 255, 0), 1);

        draw_pos = cv::Point2d(frame.points_3d[j].x * scale + drawXZ.cols / 2,
                               frame.points_3d[j].z * scale + drawXZ.rows / 2);

        cv::circle(drawXZ, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow(name + " drawXY", drawXY);
    cv::imshow(name + " drawXZ", drawXZ);

}