
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"

void BundleAdjustment::init(size_t max_frames) {
    /*
    char * argv[] = { "-v", "1" };
    int argc = sizeof(argv) / sizeof(char*);
    pba_.ParseParam(argc, argv);
*/
    pba_.SetFixedIntrinsics(true);
    pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only

    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true, 0.3, 10, 10);
    max_frames_ =  max_frames;
}




void BundleAdjustment::addKeyFrame(const VOFrame &frame, float focal, cv::Point2d pp, int feature_count){

    CameraT cam;
    cam.f = focal;

    for(int r=0; r < frame.pose_R.rows; r++){
        for(int c=0; c < frame.pose_R.cols; c++) {
            cam.m[r][c] = frame.pose_R.at<double>(r, c);
        }
    }
    for(int c=0; c < frame.pose_t.cols; c++) {
        cam.t[c] = frame.pose_t.at<double>(c);
    }

    pba_cameras_.push_back(cam);
    poses_.push_back(frame.pose.clone());

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.detectComputeORB(frame, &image_feature.keypoints, &descriptors);
    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if(features_.size() > max_frames_) {
        features_.erase(features_.begin());
        poses_.erase(poses_.begin());
    }

    (*matcher_)(features_, pairwise_matches_);

    setPBAData(features_, pairwise_matches_, poses_, &pba_point_data_, &pba_measurements_, &pba_camidx_, &pba_ptidx_);

    pba_.SetCameraData(pba_cameras_.size(), &pba_cameras_[0]);                                              //set camera parameters
    pba_.SetPointData(pba_point_data_.size(), &pba_point_data_[0]);                                         //set 3D point data
    pba_.SetProjection(pba_measurements_.size(), &pba_measurements_[0], &pba_ptidx_[0], &pba_camidx_[0]);   //set the projections*/

}

//TODO use full history rather than just the last point
int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if (!pba_.RunBundleAdjustment()) {
        LOG(INFO) << "Camera parameters adjusting failed.";
        return 1;
    }

    auto last_cam = pba_cameras_[pba_cameras_.size() - 1];

    *R = cv::Mat::eye(3, 3, CV_64FC1);
    *t = cv::Mat::zeros(3, 1, CV_64FC1);


    for(int r=0; r < R->rows; r++){
        for(int c=0; c < R->cols; c++) {
            R->at<double>(r, c) = last_cam.m[r][c];
        }
    }
    for(int c=0; c < t->cols; c++) {
        t->at<double>(c) = last_cam.t[c];
    }

    return 0;
}

//TODO This is wrong --- USe this as a template https://github.com/lab-x/SFM/blob/61bd10ab3f70a564b6c1971eaebc37211557ea78/SparseCloud.cpp
// Or this https://github.com/Zponpon/AR/blob/5d042ba18c1499bdb2ec8d5e5fae544e45c5bd91/PlanarAR/SFMUtil.cpp
void BundleAdjustment::setPBAData(const std::vector<cv::detail::ImageFeatures> &features, const std::vector<cv::detail::MatchesInfo> &pairwise_matches, const std::vector<cv::Mat> &poses,
                                                  std::vector<Point3D> *pba_point_data, std::vector<Point2D> *pba_measurements, std::vector<int> *pba_camidx, std::vector<int> *pba_ptidx){

    for(const auto & pwm : pairwise_matches){
        int idx_s = pwm.src_img_idx;
        int idx_d = pwm.dst_img_idx;
        LOG(INFO) << idx_s << " " << idx_d;

        if(idx_s != -1 && idx_d != -1) {

            std::vector<cv::Point2f> points0;
            std::vector<cv::Point2f> points1;
            for (const auto &match : pwm.matches){ //TODO use inliers mask??? Or matcher that doesnt use homography

                LOG(INFO) << match.queryIdx << " " << match.trainIdx;
                points0.push_back(features[idx_s].keypoints[match.queryIdx].pt);
                points1.push_back(features[idx_d].keypoints[match.trainIdx].pt);
                LOG(INFO) << points0[points0.size() -1] << " " << points1[points1.size() -1];

            }

            std::vector<cv::Point3f> points3d = triangulate(points0, points1, poses[idx_s], poses[idx_d]);

            for(int j = 0; j < points3d.size(); j++){ //TODO mask matched points
                Point3D p3d;

                p3d.xyz[0] = points3d[j].x;
                p3d.xyz[1] = points3d[j].y;
                p3d.xyz[2] = points3d[j].z;

                Point2D p2d;
                p2d.x = points0[j].x;
                p2d.y = points0[j].y;

                pba_point_data->push_back(p3d);
                pba_measurements->push_back(p2d);
                pba_camidx->push_back(idx_s);
                pba_ptidx->push_back(static_cast<int>(pba_measurements->size() - 1));
            }

        }
    }
}