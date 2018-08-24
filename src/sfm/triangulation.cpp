

#include "triangulation.h"

#include <cv.hpp>


void triangulate(VOFrame *prev, VOFrame *now){
    if(!now->P.empty()) {
        cv::Mat points_3d;
        cv::Mat_<double> p0(2,prev->points.size(),CV_64FC1);
        cv::Mat_<double> p1(2,now->points.size(),CV_64FC1);

        for (int i = 0; i < p0.cols; i++) {
            p0.at<double>(0, i)  = prev->points[i].x;
            p0.at<double>(1, i)  = prev->points[i].y;
            p1.at<double>(0, i)  = now->points[i].x;
            p1.at<double>(1, i)  = now->points[i].y;
        }

        cv::Mat P = cv::Mat::eye(3, 4, CV_64FC1);
        cv::triangulatePoints(P, now->P, p0, p1, points_3d); //Relative to previous frame center
        now->points_3d.clear();

        for (int i = 0; i < points_3d.cols; i++) {
            now->points_3d.push_back(cv::Point3d(points_3d.at<double>(0, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(1, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(2, i) / points_3d.at<double>(3, i)));
        }
    }
}


double getScale(const VOFrame &prev, const VOFrame &now, int num_points) {


    if (prev.points_3d.size() == 0) {
        return 1;
    }

    //Pick random points in prev that match to two points in now;
    std::vector<int> points;
    int last = -1;
    for(int i = 0; i < num_points; i++){
        for(int j = 0; j < 1000; j++){
            int index = rand() % static_cast<int>(now.points_3d.size() + 1);
            if(now.mask.at<bool>(index) && now.mask.at<bool>(prev.tracked_index[index]) && index != last) {
                last = index;
                points.push_back(index);
                break;
            }
        }
    }

    double now_sum = 0;
    double prev_sum = 0;
    for(int i = 0; i < points.size()-1; i++){
        int i0 = points[i];
        int i1 = points[i+1];
        double n0 = cv::norm(now.points_3d[i0] - now.points_3d[i1]);
        double n1 = cv::norm(prev.points_3d[prev.tracked_index[i0]] - prev.points_3d[prev.tracked_index[i1]]);

        if(!std::isnan(n0) && !std::isnan(n1) && !std::isinf(n0) && !std::isinf(n1)) {
            now_sum += n0;
            prev_sum += n1;
        }
    }

    return  now_sum / prev_sum ;
}

