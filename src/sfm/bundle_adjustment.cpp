
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"
#include <opencv2/sfm/triangulation.hpp>
#include <iostream>

#include "src/utils/draw.h"

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>




void BundleAdjustment::init(const cv::Mat &K, size_t max_frames) {

    max_frames_ = max_frames;

    K_ = K.clone();

    focal_ = cv::Point2d(K.at<double>(0,0), K.at<double>(1,1));
    pp_ = cv::Point2d(K.at<double>(0,2), K.at<double>(1,2));

    //sba_.
}


void BundleAdjustment::matcher() {
    const float ratio = 0.8; // As in Lowe's paper; can be tuned

    cv::BFMatcher matcher;
    pairwise_matches_.clear();
    for (int i = 0; i < features_.size() - 1; i++) {
        int j = i + 1;

        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(features_[i].descriptors, features_[j].descriptors, matches, 2);  // Find two nearest matches

        cv::detail::MatchesInfo good_matches;

        std::vector<cv::Point2f> points0;
        std::vector<cv::Point2f> points1;
        for (int k = 0; k < matches.size(); k++) {

            if (matches[k][0].distance < ratio * matches[k][1].distance) {

                const auto &p0 = features_[i].keypoints[matches[k][0].queryIdx];
                const auto &p1 = features_[j].keypoints[matches[k][0].trainIdx];

                if (cv::norm(cv::Mat(p0.pt) - cv::Mat(p1.pt)) < 100) {
                    good_matches.matches.push_back(matches[k][0]);
                    good_matches.inliers_mask.push_back(1);

                    points0.push_back(p0.pt);
                    points1.push_back(p1.pt);
                }

            }
        }
        cv::Mat mask, R, t;

        if(points0.size()>= 5){
            cv::Mat E = cv::findEssentialMat(points0, points1, K_, cv::RANSAC, 0.999, 1.0, mask);

            for (int k = 0; k < good_matches.inliers_mask.size(); k++) {
                if (!mask.at<bool>(k)) {
                    good_matches.inliers_mask[k] = 0;
                }
            }

            good_matches.src_img_idx = i;
            good_matches.dst_img_idx = j;

            pairwise_matches_.push_back(good_matches);
        }
    }

    createTracks();
}

void BundleAdjustment::createTracks() {

    tracks_.clear();
    std::vector<std::unordered_map<int, int>> pairs(camera_matrix_.size() - 1);
    for (auto &pwm : pairwise_matches_) {
        int idx_cam0 = pwm.src_img_idx;
        for (int i = 0; i < pwm.matches.size(); i++) {
            auto &match = pwm.matches[i];

            if (pwm.inliers_mask[i] != 0) {
                pairs[idx_cam0][match.queryIdx] = match.trainIdx;
            }
        }
    }

    tracks_.resize(camera_matrix_.size() - 1);

    for (int cam_idx = 0; cam_idx < pairs.size(); cam_idx++) {

        for (std::pair<int, int> element : pairs[cam_idx]) {

            if (element.second == -1) {
                continue;
            }

            std::vector<int> track;
            track.push_back(element.first);
            track.push_back(element.second);
            int key = element.second;
            int cam = cam_idx + 1;

            if (cam < pairs.size()) {
                while (pairs[cam].find(key) != pairs[cam].end()) {
                    key = pairs[cam][key];
                    if (key == -1) {
                        break;
                    }

                    pairs[cam][key] = -1; //seen
                    track.push_back(key);
                    cam++;

                    if (cam == pairs.size()) {
                        break;
                    }
                }
            }
            tracks_[cam_idx].push_back(track);
        }
    }
}


//// THIS IS THE PROBLEM http://www.land-of-kain.de/docs/coords/
void BundleAdjustment::addKeyFrame(const VOFrame &frame) {


    camera_matrix_.push_back(K_.clone());
    R_.push_back(frame.pose_R.clone());
    T_.push_back(frame.pose_t.clone());
    dist_coeffs_.push_back(cv::Mat::zeros(5,1,CV_64F));

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.computeORB(frame, &image_feature.keypoints, &descriptors);
    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        camera_matrix_.erase(camera_matrix_.begin());
        R_.erase(R_.begin());
        T_.erase(T_.begin());
        dist_coeffs_.erase(dist_coeffs_.begin());
    }

    matcher();

    setPBAPoints();
}

// TODO This is wrong --- Use this as a template https://github.com/lab-x/SFM/blob/61bd10ab3f70a564b6c1971eaebc37211557ea78/SparseCloud.cpp
// Or this https://github.com/Zponpon/AR/blob/5d042ba18c1499bdb2ec8d5e5fae544e45c5bd91/PlanarAR/SFMUtil.cpp
// https://stackoverflow.com/questions/46875340/parallel-bundle-adjustment-pba
void BundleAdjustment::setPBAPoints() {

    points_3d_.clear();
    points_img_.clear();
    visibility_.clear();

    visibility_.resize(camera_matrix_.size());
    points_img_.resize(camera_matrix_.size());

    std::vector<cv::Mat> poses;
    for (int i  =0; i < R_.size(); i++) {
        cv::Mat P;
        hconcat(R_[i], T_[i], P);
        poses.push_back(std::move(P));
    }

    cv::Mat tracks = cv::Mat::zeros(pp_.y * 2, pp_.x * 2, CV_8UC3);

    for (int cam_idx = 0; cam_idx < tracks_.size(); cam_idx++) {

        for (const auto &track : tracks_[cam_idx]) {

            std::vector<cv::Point2f> points;
            for (int i = 0; i < track.size(); i++) {
                points.push_back(features_[cam_idx + i].keypoints[track[i]].pt);
            }

            if (points.size() < 2) {
                continue;
            }

            std::vector<cv::Point2f> points0 = {points[0]};
            std::vector<cv::Point2f> points1 = {points[1]};

            std::vector<cv::Mat> sfm_points_2d;
            std::vector<cv::Mat> projection_matrices;
            double dist = 0;
            cv::Mat p, p_up;
            std::vector<cv::Point3d> points3d;


            for (int i = 0; i < points.size(); i++) {
                sfm_points_2d.emplace_back((cv::Mat_<double>(2, 1) << points[i].x, points[i].y));
                projection_matrices.push_back(K_ * poses[cam_idx + i]);
            }

            cv::Mat point_3d_mat;
            cv::sfm::triangulatePoints(sfm_points_2d, projection_matrices, point_3d_mat);
            points3d = points3DToVec(point_3d_mat);

            p = cv::Mat(points3d[0]);

            p_up = (R_[cam_idx].t() * p) - T_[cam_idx];
            dist = 0;//cv::norm(p_up);


            //TODO make sure z is pos
            if (dist < kMax3DDist) {// && p_up.at<double>(0,2) < 0) { //TODO why are points wrong when I draw them

                points_3d_.push_back( points3d[0]);

                std::vector< cv::Point2d > points_img(camera_matrix_.size(), cv::Point2d(0,0));
                std::vector< int > visibility(camera_matrix_.size(), 0);

                for (int i = 0; i < points.size(); i++) {

                    //LOG(INFO) << cam_idx + i << " " << i;
                    //reprojectionInfo(points[i], points3d[0], poses[cam_idx + i]); //TODO For info - remove later

                    points_img[cam_idx + i] = points[i];
                    visibility[cam_idx + i] = 1;

                    if (i < points.size() - 1) {
                        cv::line(tracks, points[i], points[i + 1], cv::Scalar(0, 255, 0), 1);
                        cv::circle(tracks, points[i], 2, cv::Scalar(255, 0, 0), 2);
                    }
                }



                for(int i=0; i< camera_matrix_.size(); i++) {
                    visibility_[i].push_back(visibility[i]);
                    points_img_[i].push_back(points_img[i]);

                }
            }
        }
    }

    imshow("tracks", tracks);
    LOG(INFO) << points_3d_.size() << " " << points_img_.size();
}

//TODO use full history rather than just updating the newest point
int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if (points_3d_.empty()) {
        LOG(INFO) << "Bundle adjustment points are empty";
        return 1;
    }


    sba_.run(points_3d_, points_img_, visibility_, camera_matrix_, R_, T_, dist_coeffs_);

    LOG(INFO) <<"Initial error="<<sba_.getInitialReprjError()<<". "<<
             "Final error="<<sba_.getFinalReprjError();

    *R = R_[R_.size()-1];

    *t = T_[T_.size()-1];


    return 0;
}

void BundleAdjustment::draw(float scale) {
    cv::Mat ba_map(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(ba_map, cv::Point(ba_map.cols / 2, 0), cv::Point(ba_map.cols / 2, ba_map.rows), cv::Scalar(0, 0, 255));
    cv::line(ba_map, cv::Point(0, ba_map.rows / 2), cv::Point(ba_map.cols, ba_map.rows / 2), cv::Scalar(0, 0, 255));

    for (const auto &p : points_3d_) {
        cv::Point2d draw_pos = cv::Point2d(p.x * scale + ba_map.cols / 2, p.z * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    for (int i  =0; i < T_.size(); i++) {

        const cv::Mat & t = T_[i];
        cv::Point2d draw_pos = cv::Point2d(t.at<double>(0) * scale + ba_map.cols / 2, t.at<double>(2) * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(255, 0, 0), 2);

        const cv::Mat &R = R_[i];
        double data[3] = {0, 0, 1};
        cv::Mat dir(3, 1, CV_64FC1, data);

        dir = (R * dir) * 10 * scale;

        cv::line(ba_map, draw_pos, cv::Point2d(dir.at<double>(0, 0), dir.at<double>(0, 2)) + draw_pos,
                 cv::Scalar(0, 255, 255), 2);

    }

    if (!T_.empty()) {
        const auto &t = T_[T_.size() - 1];
        cv::Point2d draw_pos = cv::Point2d(t.at<double>(0) * scale + ba_map.cols / 2, t.at<double>(2) * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("BA_Map", ba_map);
}



void BundleAdjustment::drawViz(){

    cv::viz::Viz3d myWindow("Coordinate Frame");
    myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    int count = 0;
    for (int i  =0; i < T_.size(); i++) {

        const cv::Mat & t = T_[i];
        const cv::Mat & R = R_[i];

        auto col = cv::viz::Color::red();
        if(count % 3 == 1) {
            col = cv::viz::Color::green();
        } else if(count % 3 == 2) {
            col = cv::viz::Color::blue();
        }

        cv::viz::WCameraPosition cam(cv::Matx33d(K_), 3, col);
        myWindow.showWidget("c" + std::to_string(count), cam);

        cv::Affine3d pose(R, t);
        myWindow.setWidgetPose("c" + std::to_string(count), pose);
        count++;
    }
    cv::viz::WCloud cloud_widget1(points_3d_, cv::viz::Color::green());

    //myWindow.showWidget("cloud 2", cloud_widget1);

    myWindow.spin();
}
