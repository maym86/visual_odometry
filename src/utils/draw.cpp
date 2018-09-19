

#include "draw.h"



void drawPosePoint(float scale, const cv::Mat &P, cv::Mat *drawXY, cv::Mat *drawXZ){

    cv::Mat R = P.colRange(cv::Range(0, 3));
    cv::Mat t = scale * P.col(3);

    cv::Point2d posXY( t.at<double>(0) + drawXY->cols / 2, t.at<double>(1) + drawXY->rows / 2);

    cv::circle(*drawXY, posXY, 2, cv::Scalar(0, 0, 255), 2);

    cv::Point2d posXZ( t.at<double>(0) + drawXZ->cols / 2, t.at<double>(2) + drawXZ->rows / 2);

    cv::circle(*drawXZ, posXZ, 2, cv::Scalar(0, 0, 255), 2);


    double dataXY[3] = {0,1,0};
    cv::Mat dirXY(3,1,CV_64FC1, dataXY);
    dirXY = (R * dirXY) * 5 * scale;
    //cv::line(*drawXY, posXY, cv::Point2d(dirXY.at<double>(0,0), dirXY.at<double>(0,1) ) + posXY, cv::Scalar(0,255,255), 2 );


    double dataXZ[3] = {0,0,1};
    cv::Mat dirXZ(3,1,CV_64FC1, dataXZ);
    dirXZ = (R * dirXZ) * 5 * scale;
    cv::line(*drawXZ, posXZ, cv::Point2d(dirXZ.at<double>(0,0), dirXZ.at<double>(0,2) ) + posXZ, cv::Scalar(0,255,255), 2 );

}

void draw3D(const std::string &name, std::vector<cv::Point3d> &points_3d, float scale, const cv::Mat &P0, const cv::Mat &P1){
    cv::Mat drawXY(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat drawXZ(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(drawXY, cv::Point(drawXY.cols / 2, 0), cv::Point(drawXY.cols / 2, drawXY.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXY, cv::Point(0, drawXY.rows / 2), cv::Point(drawXY.cols, drawXY.rows / 2), cv::Scalar(0, 0, 255));

    cv::line(drawXZ, cv::Point(drawXZ.cols / 2, 0), cv::Point(drawXZ.cols / 2, drawXZ.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXZ, cv::Point(0, drawXZ.rows / 2), cv::Point(drawXZ.cols, drawXZ.rows / 2), cv::Scalar(0, 0, 255));

    for (int j = 0; j < points_3d.size(); j++) {
        cv::Point2d draw_pos = cv::Point2d(points_3d[j].x * scale + drawXY.cols / 2,
                               points_3d[j].y * scale + drawXY.rows / 2);

        cv::circle(drawXY, draw_pos, 1, cv::Scalar(0, 255, 0), 1);

        draw_pos = cv::Point2d(points_3d[j].x * scale + drawXZ.cols / 2,
                               points_3d[j].z * scale + drawXZ.rows / 2);

        cv::circle(drawXZ, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    drawPosePoint(scale, P0, &drawXY, &drawXZ);
    drawPosePoint(scale, P1, &drawXY, &drawXZ);


    cv::imshow(name + " drawXY", drawXY);
    cv::imshow(name + " drawXZ", drawXZ);
}


void drawPose(const std::string &name, std::vector<VOFrame> &frame, float scale){
    cv::Mat drawXZ(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(drawXZ, cv::Point(drawXZ.cols / 2, 0), cv::Point(drawXZ.cols / 2, drawXZ.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXZ, cv::Point(0, drawXZ.rows / 2), cv::Point(drawXZ.cols, drawXZ.rows / 2), cv::Scalar(0, 0, 255));

    for (int j = 0; j < frame.size(); j++) {

        cv::Point2d pos(scale * frame[j].pose.at<double>(0, 3) + drawXZ.cols / 2,
                        scale * frame[j].pose.at<double>(2, 3) + drawXZ.rows / 2);



        double data[3] = {0,0,1};
        cv::Mat dir(3,1,CV_64FC1, data);

        cv::Mat R = frame[j].pose.colRange(cv::Range(0, 3));

        dir = (R * dir) * 10 * scale;

        cv::line(drawXZ, pos, cv::Point2d(dir.at<double>(0,0), dir.at<double>(0,2) ) + pos, cv::Scalar(0,255,255), 2 );

        cv::circle(drawXZ, pos, 1, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(name + " drawXZ", drawXZ);

}


void drawMatches(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::Point2f> &p0, std::vector<cv::Point2f> &p1) {

    cv::Mat output = image.clone();
    for (int i = 0; i < p0.size(); i++) {
        //if (mask.at<bool>(i)) {
            cv::circle(output, p0[i], 2, cv::Scalar(255,0,0), 2);
            cv::line(output, p0[i], p1[i], cv::Scalar(0,255,0), 2);
        //}
    }

    cv::imshow("Matches", output);

}
