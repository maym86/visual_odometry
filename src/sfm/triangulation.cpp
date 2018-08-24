

#include "triangulation.h"

#include <cv.hpp>


void triangulate(VOFrame *vo0, VOFrame *vo1){
    if(!vo1->P.empty()) {
        cv::Mat points_3d;
        cv::Mat_<double> p0(2,vo0->points.size(),CV_64FC1);
        cv::Mat_<double> p1(2,vo1->points.size(),CV_64FC1);

        for (int i = 0; i < p0.cols; i++) {
            p0.at<double>(0, i)  = vo0->points[i].x;
            p0.at<double>(1, i)  = vo0->points[i].y;
            p1.at<double>(0, i)  = vo1->points[i].x;
            p1.at<double>(1, i)  = vo1->points[i].y;
        }

        cv::Mat P = cv::Mat::eye(3, 4, CV_64FC1);
        cv::triangulatePoints(P, vo1->P, p0, p1, points_3d); //TODO this is relative to previous frame center
        vo1->points_3d.clear();

        for (int i = 0; i < points_3d.cols; i++) {
            vo1->points_3d.push_back(cv::Point3d(points_3d.at<double>(0, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(1, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(2, i) / points_3d.at<double>(3, i)));
        }
    }
}


double getScale(const VOFrame &vo0, const VOFrame &vo1, int num_points) {


    if (vo0.points_3d.size() == 0 || vo1.points_3d.size() == 0) {
        return 1;
    }

    //Pick random points in prev that match to two points in now;
    std::vector<int> points;
    int last = -1;
    for(int i = 0; i < num_points; i++){
        for(int j = 0; j < 1000; j++){
            int index = rand() % static_cast<int>(vo1.points_3d.size() + 1);
            if(vo1.mask.at<bool>(index) && vo1.mask.at<bool>(vo0.tracked_index[index]) && index != last) {
                last = index;
                points.push_back(index);
                break;
            }
        }
    }

    if(points.empty()) {
        return 1;
    }

    double now_sum = 0;
    double prev_sum = 0;
    for(int i = 0; i < points.size()-1; i++){
        int i0 = points[i];
        int i1 = points[i+1];
        double n0 = cv::norm(vo1.points_3d[i0] - vo1.points_3d[i1]);
        double n1 = cv::norm(vo0.points_3d[vo0.tracked_index[i0]] - vo0.points_3d[vo0.tracked_index[i1]]);


        if(!std::isnan(n0) && !std::isnan(n1) && !std::isinf(n0) && !std::isinf(n1) && abs(n1-n0) < 50) {
            now_sum += n0;
            prev_sum += n1;
        }
    }

    double scale = now_sum / prev_sum;
    if(std::isnan(scale) || std::isnan(scale) || scale == 0)
        return 1;

    return scale;
}

