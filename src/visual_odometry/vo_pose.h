
#ifndef VO_VO_POSE_H
#define VO_VO_POSE_H

#include "src/sfm/triangulation.h"
#include <glog/logging.h>
#include "src/utils/draw.h"

const size_t kMinPosePoints = 8;
const float kMax3DDist = 200;

inline int updatePose(const cv::Mat &K, VOFrame *last_frame, VOFrame *new_frame) {

    new_frame->E = cv::findEssentialMat(last_frame->points, new_frame->points, K, cv::LMEDS, 0.999, 1.0,
                                        new_frame->mask);


    cv::Mat points_4d;
    int res = recoverPose(new_frame->E, last_frame->points, new_frame->points, K, new_frame->local_R,
                          new_frame->local_t, kMax3DDist, new_frame->mask, points_4d);

    new_frame->points_3d = points4DToVec(points_4d);


    if (res > kMinPosePoints) {

        // https://stackoverflow.com/questions/37810218/is-the-recoverpose-function-in-opencv-is-left-handed
        // http://www.land-of-kain.de/docs/coords/
        new_frame->local_t = -new_frame->local_t;
        new_frame->local_R = new_frame->local_R.t();

        hconcat(new_frame->local_R, new_frame->local_t, new_frame->local_pose);


        new_frame->scale = getScale(*last_frame, *new_frame, kMinPosePoints, 200, kMax3DDist);

        new_frame->pose_t = last_frame->pose_t + new_frame->scale * (last_frame->pose_R * new_frame->local_t);
        new_frame->pose_R = new_frame->local_R * last_frame->pose_R;

        hconcat(new_frame->pose_R, new_frame->pose_t, new_frame->pose);

        return 0;
    } else {
        //Copy last pose
        LOG(INFO) << "RecoverPose, too few points";
        new_frame->pose_R = last_frame->pose_R.clone();
        new_frame->pose_t = last_frame->pose_t.clone();
        new_frame->pose = last_frame->pose.clone();

        return 1;
    }

}

#endif //VO_VO_POSE_H
