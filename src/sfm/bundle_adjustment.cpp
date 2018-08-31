
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"

void BundleAdjustment::init(size_t max_frames) {

    ParallelBA::DeviceT device = ParallelBA::PBA_CUDA_DEVICE_DEFAULT; //device = ParallelBA::PBA_CPU_DOUBLE;
    pba_ = ParallelBA(device);
    //pba_.SetFixedIntrinsics(true);
    //pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only

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

    std::vector<cv::Point2f> temp;
    for (const auto &kp : image_feature.keypoints){
        temp.push_back(kp.pt);
    }
    keypoints_.push_back(std::move(temp));


    if(features_.size() > max_frames_) {
        features_.erase(features_.begin());
        poses_.erase(poses_.begin());
        keypoints_.erase(keypoints_.begin());
    }

    (*matcher_)(features_, pairwise_matches_);

    if(poses_.size() > 1) {
         //setPBAData(keypoints_, pairwise_matches_, poses_, &pba_point_data_, &pba_measurements_, &pba_camidx_, &pba_ptidx_);

        //pba_.SetCameraData(pba_cameras_.size(), &pba_cameras_[0]);                                              //set camera parameters
        //pba_.SetPointData(pba_point_data_.size(), &pba_point_data_[0]);                                         //set 3D point data
        //pba_.SetProjection(pba_measurements_.size(), &pba_measurements_[0], &pba_ptidx_[0], &pba_camidx_[0]);   //set the projections*/
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

//TODO This is wrong
void BundleAdjustment::setPBAData(const std::vector<std::vector<cv::Point2f>> &keypoints, const std::vector<cv::detail::MatchesInfo> &pairwise_matches, const std::vector<cv::Mat> &poses,
                                                  std::vector<Point3D> *pba_point_data, std::vector<Point2D> *pba_measurements, std::vector<int> *pba_camidx, std::vector<int> *pba_ptidx){

    for(int i = 0; i < pairwise_matches.size(); i++){
        const auto &pwm = pairwise_matches[i];

        int idx_s = pwm.src_img_idx;
        int idx_d = pwm.dst_img_idx;

        if(idx_s != -1 && idx_d != -1) {
            std::vector<cv::Point3f> points3d = triangulate(keypoints[idx_s], keypoints[idx_d], poses[idx_s], poses[idx_d]);

            for(int j = 0; j < points3d.size(); j++){
                Point3D p3d;
                p3d.xyz[0] = points3d[j].x;
                p3d.xyz[1] = points3d[j].y;
                p3d.xyz[2] = points3d[j].z;

                Point2D p2d;
                p2d.x = keypoints[idx_s][j].x;
                p2d.y = keypoints[idx_s][j].y;

                pba_point_data->push_back(p3d);
                pba_measurements->push_back(p2d);
                pba_camidx->push_back(idx_s);
                pba_ptidx->push_back(j);
            }
        }
    }
}