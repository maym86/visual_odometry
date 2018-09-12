

#ifndef VO_DRAW_H
#define VO_DRAW_H


#include "src/visual_odometry/vo_frame.h"

void draw3D(const std::string &name, std::vector<cv::Point3d> &points_3d, float scale);


void drawPose(const std::string &name, std::vector<VOFrame> &frame, float scale);

void drawMatches(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::Point2f> &p0, std::vector<cv::Point2f> &p1);

#endif //VO_DRAW_H
