
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"

void BundleAdjustment::init(size_t max_frames) {

    ParallelBA::DeviceT device = ParallelBA::PBA_CUDA_DEVICE_DEFAULT; //device = ParallelBA::PBA_CPU_DOUBLE;
    pba_ = ParallelBA(device);
    //pba_.SetFixedIntrinsics(true);
    //pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only

    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true, 0.2, 10, 10);
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
    image_feature.img_idx = count_;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    std::vector<cv::Point2f> temp;
    for (const auto &kp : image_feature.keypoints){
        temp.push_back(kp.pt);
    }
    keypoints_.push_back(temp);


    if(features_.size() > max_frames_) {
        features_.erase(features_.begin());
        poses_.erase(poses_.begin());
        keypoints_.erase(keypoints_.begin());
    }


    //Do pairwise matching first
    (*matcher_)(features_, pairwise_matches_);
    count_++;

    if(poses_.size() > 1) {
        triangulatePairwiseMatches(keypoints_, pairwise_matches_, poses_, &pba_point_data_, &pba_measurements_, &pba_camidx_, &pba_ptidx_);

        // TODO figure out how to set this
        pba_.SetCameraData(pba_cameras_.size(), &pba_cameras_[0]);                                           //set camera parameters
        pba_.SetPointData(pba_point_data_.size(), &pba_point_data_[0]);                                         //set 3D point data
        //pba_.SetProjection(pba_measurements_.size(), &pba_measurements_[0], &pba_ptidx_[0], &pba_camidx_[0]);  //set the projections
    }
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