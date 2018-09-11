
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"


#include "src/utils/draw.h"


BundleAdjustment::BundleAdjustment() : pba_(ParallelBA::DeviceT::PBA_CPU_DOUBLE) {

}

void BundleAdjustment::init(const cv::Point2f &focal, const cv::Point2f &pp, size_t max_frames) {

    char *argv[] = {"-lmi<100>", "-v", "1"};
    int argc = sizeof(argv) / sizeof(char *);

    pba_.ParseParam(argc, argv);

    pba_.SetFixedIntrinsics(true);

    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true);
    max_frames_ = max_frames;

    pp_ = pp;
    focal_ = focal;
}


void BundleAdjustment::matcher() {
    cv::BFMatcher matcher;
    pairwise_matches_.clear();
    for(int i =  0; i < features_.size(); i++) {
        for(int j=0; j < features_.size(); j++) {

            if(abs(i -j) > 2){
                continue;
            }

            std::vector<std::vector<cv::DMatch>> matches;
            matcher.knnMatch(features_[i].descriptors, features_[j].descriptors, matches, 2);  // Find two nearest matches

            cv::detail::MatchesInfo good_matches;
            for (int k = 0; k < matches.size(); ++k) {
                const float ratio = 0.8; // As in Lowe's paper; can be tuned
                if (matches[k][0].distance < ratio * matches[k][1].distance && matches[k][0].distance > 200) {


                    auto p0 = features_[i].keypoints[matches[k][0].queryIdx];
                    auto p1 = features_[j].keypoints[matches[k][0].trainIdx];

                    if(cv::norm(cv::Mat(p0.pt) - cv::Mat(p1.pt)) < 50) {
                        good_matches.matches.push_back(matches[k][0]);
                        good_matches.inliers_mask.push_back(1);
                    }
                }
            }

            good_matches.src_img_idx = i;
            good_matches.dst_img_idx = j;

            if(good_matches.matches.size() > 10) {
                pairwise_matches_.push_back(good_matches);
            }
        }
    }

}

void BundleAdjustment::addKeyFrame(const VOFrame &frame) {

    CameraT cam;
    cam.f = focal_.x;

    cam.SetTranslation(reinterpret_cast<double *>(frame.pose_t.data));
    cam.SetMatrixRotation(reinterpret_cast<double *>(frame.pose_R.data));

    pba_cameras_.push_back(cam);

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.detectComputeORB(frame, &image_feature.keypoints, &descriptors);
    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        pba_cameras_.erase(pba_cameras_.begin());
    }

    pba_cameras_[0].SetConstantCamera();

    //matcher();
    (*matcher_)(features_, pairwise_matches_);

    setPBAPoints();

}

// TODO This is wrong --- Use this as a template https://github.com/lab-x/SFM/blob/61bd10ab3f70a564b6c1971eaebc37211557ea78/SparseCloud.cpp
// Or this https://github.com/Zponpon/AR/blob/5d042ba18c1499bdb2ec8d5e5fae544e45c5bd91/PlanarAR/SFMUtil.cpp
// https://stackoverflow.com/questions/46875340/parallel-bundle-adjustment-pba
void BundleAdjustment::setPBAPoints() {

    pba_3d_points_.clear();
    pba_image_points_.clear();
    pba_cam_idx_.clear();
    pba_2d3d_idx_.clear();

    for (const auto &pwm : pairwise_matches_) {
        int idx_cam0 = pwm.src_img_idx;
        int idx_cam1 = pwm.dst_img_idx;

        if (idx_cam0 != -1 && idx_cam1 != -1 && idx_cam0 != idx_cam1) { //TODO experiment with confidence thresh

            std::vector<cv::Point2f> points0;
            std::vector<cv::Point2f> points1;

            for (int i = 0; i < pwm.matches.size(); i++) {
                const auto &match = pwm.matches[i];
                if (pwm.inliers_mask[i]) {
                    points0.push_back(features_[idx_cam0].keypoints[match.queryIdx].pt);
                    points1.push_back(features_[idx_cam1].keypoints[match.trainIdx].pt);
                }
            }

            cv::Mat matches = cv::Mat::zeros(376, 1241, CV_8UC3);
            drawMatches(matches, cv::Mat() , points0, points1);

            cv::Mat t0 = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat R0 = cv::Mat::eye(3, 3, CV_64FC1);

            pba_cameras_[idx_cam0].GetTranslation(reinterpret_cast<double *>(t0.data));
            pba_cameras_[idx_cam0].GetMatrixRotation(reinterpret_cast<double *>(R0.data));

            cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat R1 = cv::Mat::eye(3, 3, CV_64FC1);

            pba_cameras_[idx_cam1].GetTranslation(reinterpret_cast<double *>(t1.data));
            pba_cameras_[idx_cam1].GetMatrixRotation(reinterpret_cast<double *>(R1.data));

            cv::Mat P0; //= cv::Mat::eye(3, 4, CV_64FC1);
            hconcat(R0, t0, P0);
            cv::Mat P1;
            hconcat(R1, t1, P1);

            //Remove P0 offset
            //P1.col(3) -= t0;
            //P1.colRange(cv::Range(0, 3)) *= R0.t();

            std::vector<cv::Point3d> points3d = triangulate(pp_, focal_, points0, points1, P0, P1);

            for (int j = 0; j < points3d.size(); j++) {

                cv::Mat p = cv::Mat(points3d[j]);
                double dist = cv::norm(p - t0);

                if (dist < kMax3DDist) {//&& points3d[j].z > 0) {
                    //p = (R0 * p.t()) + t0;
                    pba_3d_points_.emplace_back(Point3D{static_cast<float>(p.at<double>(0, 0)),
                                                        static_cast<float>(p.at<double>(0, 1)),
                                                        static_cast<float>(p.at<double>(0, 2))});

                    //First 2dpoint that relates to 3d point
                    pba_image_points_.emplace_back(Point2D{points0[j].x - pp_.x, points0[j].y - pp_.y});
                    pba_cam_idx_.push_back(idx_cam0);
                    pba_2d3d_idx_.push_back(static_cast<int>(pba_3d_points_.size() - 1));

                    //Second 2dpoint that relates to 3D point
                    pba_image_points_.emplace_back(Point2D{points1[j].x - pp_.x, points1[j].y - pp_.y});
                    pba_cam_idx_.push_back(idx_cam1);
                    pba_2d3d_idx_.push_back(static_cast<int>(pba_3d_points_.size() - 1));
                }
            }
        }
    }

    LOG(INFO) << pba_3d_points_.size() << " " << pba_image_points_.size();
}

void BundleAdjustment::draw(float scale){
    cv::Mat ba_map(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(ba_map, cv::Point(ba_map.cols / 2, 0), cv::Point(ba_map.cols / 2, ba_map.rows), cv::Scalar(0, 0, 255));
    cv::line(ba_map, cv::Point(0, ba_map.rows / 1.5), cv::Point(ba_map.cols, ba_map.rows / 1.5), cv::Scalar(0, 0, 255));

    for (const auto &p : pba_3d_points_) {
        cv::Point2d draw_pos = cv::Point2d(p.xyz[0] * scale + ba_map.cols / 2, p.xyz[2] * scale + ba_map.rows / 1.5);
        cv::circle(ba_map, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    for (const auto &cam : pba_cameras_){
        cv::Point2d draw_pos = cv::Point2d(cam.t[0] * scale + ba_map.cols / 2, cam.t[2] * scale + ba_map.rows / 1.5);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(255, 0, 0), 2);
    }

    if(!pba_cameras_.empty()) {
        const auto &cam = pba_cameras_[pba_cameras_.size() - 1];
        cv::Point2d draw_pos = cv::Point2d(cam.t[0] * scale + ba_map.cols / 2, cam.t[2] * scale + ba_map.rows / 1.5);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("BA_Map", ba_map);
}


//TODO use full history rather than just updating the newest point
int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if(pba_3d_points_.empty() || pba_image_points_.empty()) {
        LOG(INFO) << "Bundle adjustment points are empty";
        return 1;
    }

    pba_.SetCameraData(pba_cameras_.size(), &pba_cameras_[0]); //set camera parameters
    pba_.SetPointData(pba_3d_points_.size(), &pba_3d_points_[0]); //set 3D point data

    //set the projections
    pba_.SetProjection(pba_image_points_.size(), &pba_image_points_[0], &pba_2d3d_idx_[0], &pba_cam_idx_[0]);
    pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only

    if (!pba_.RunBundleAdjustment()) {
        LOG(INFO) << "Camera parameters adjusting failed.";
        return 1;
    }

    const auto &last_cam = pba_cameras_[pba_cameras_.size() - 1];

    *R = cv::Mat::eye(3, 3, CV_64FC1);
    *t = cv::Mat::zeros(3, 1, CV_64FC1);

    last_cam.GetTranslation(reinterpret_cast<double *>(t->data));
    last_cam.GetMatrixRotation(reinterpret_cast<double *>(R->data));

    return 0;
}
