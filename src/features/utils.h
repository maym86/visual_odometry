
#ifndef VO_FETAURES_UTILS_H
#define VO_FETAURES_UTILS_H

#include <vector>
#include "src/visual_odometry/vo_frame.h"

void removePoints(VOFrame *vo0, VOFrame *vo1, std::vector<unsigned char> *status){

    //Remove bad points
    for (int i = status->size() - 1; i >= 0; --i) {
        cv::Point2f pt = vo1->points[i];
        if ((*status)[i] == 0 || pt.x < 0 || pt.y < 0) {
            if (pt.x < 0 || pt.y < 0) {
                (*status)[i] = 0;
            }
            vo0->points.erase(vo0->points.begin() + i);
            vo1->points.erase(vo1->points.begin() + i);
        }
        else{
            vo0->tracked_index.push_back(i); //indices of points that are kept
        }
    }
    std::reverse(vo0->tracked_index.begin(),vo0->tracked_index.end());

};



#endif //VO_FETAURES_UTILS_H
