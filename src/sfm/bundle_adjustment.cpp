
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"
#include <opencv2/sfm/triangulation.hpp>
#include <iostream>

#include "src/utils/draw.h"

#include <opencv2/core/eigen.hpp>


void BundleAdjustment::init(const cv::Mat &K, size_t max_frames) {

    max_frames_ = max_frames;

    K_ = K.clone();

    focal_ = cv::Point2d(K.at<double>(0,0), K.at<double>(1,1));
    pp_ = cv::Point2d(K.at<double>(0,2), K.at<double>(1,2));

    viz_ = cv::viz::Viz3d("Coordinate Frame");
    viz_.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
}


void BundleAdjustment::matcher() {
    const float ratio = 0.8; // As in Lowe's paper; can be tuned


    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //cv::BFMatcher matcher;
    pairwise_matches_.clear();
    for (int i = 0; i < features_.size() - 1; i++) {
        for (int j = i+1; j < std::min((int)features_.size(),i+3) ; j++) {

            std::vector<std::vector<cv::DMatch>> matches;
            matcher->knnMatch(features_[i].descriptors, features_[j].descriptors, matches,
                              2);  // Find two nearest matches

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

            if (points0.size() >= 5) {
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
    }

    createTracks();
}

void BundleAdjustment::createTracks() {

    tracks_.clear();
    std::vector<std::unordered_map<int, std::pair<int,int>>> pairs(camera_matrix_.size());
    for (auto &pwm : pairwise_matches_) {
        int idx_cam0 = pwm.src_img_idx;
        for (int i = 0; i < pwm.matches.size(); i++) {
            auto &match = pwm.matches[i];

            if (pwm.inliers_mask[i] != 0) { //THIS is writing over matches
                pairs[idx_cam0][match.queryIdx] = std::pair<int,int>(pwm.dst_img_idx, match.trainIdx);
            }
        }
    }
    //TODO validate this
    tracks_.resize(camera_matrix_.size() - 1);


    std::vector<std::vector<int>> landmarks_; //array of matched features per camera // [kp_idx][poses] -1 == not present in image


    for (int cam_idx = 0; cam_idx < pairs.size(); cam_idx++) {

        for (std::pair<int, std::pair<int,int>> element : pairs[cam_idx]) {

            if (element.second.second == -1) {
                continue;
            }

            std::vector<std::pair<int,int>> track;
            track.push_back(std::pair<int,int>(cam_idx,element.first));
            track.push_back(element.second);

            int next_cam = element.second.first;
            int next_key = element.second.second;

            LOG(INFO) << next_cam << " " << next_key;

            while (pairs[next_cam].find(next_key) != pairs[next_cam].end()) {


                auto val = pairs[next_cam][next_key];
                if (val.second == -1) {
                    break;
                }

                pairs[next_cam][next_key].second = -1; //seen
                track.push_back(val);
                next_cam = val.first;
                next_key = val.second;

            }

            tracks_[cam_idx].push_back(track);
            LOG(INFO) << cam_idx << " " << tracks_[cam_idx].size();
        }
    }
}

void BundleAdjustment::addKeyFrame(const VOFrame &frame) {

    camera_matrix_.push_back(K_.clone());
    R_.push_back(frame.pose_R.clone());
    t_.push_back(frame.pose_t.clone());
    dist_coeffs_.push_back(cv::Mat::zeros(5,1,CV_64F));

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.detectComputeAKAZE(frame, &image_feature.keypoints, &descriptors);

    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        camera_matrix_.erase(camera_matrix_.begin());
        R_.erase(R_.begin());
        t_.erase(t_.begin());
        dist_coeffs_.erase(dist_coeffs_.begin());
    }

    matcher();

    //TODO use pairwise matcher and then updte create tracks to work with any to any matches
    //Create match matrix

    setPBAPoints();
}

// TODO This is wrong - the triangualtion results are weird
void BundleAdjustment::setPBAPoints() {

    points_3d_.clear();
    points_img_.clear();
    visibility_.clear();

    visibility_.resize(camera_matrix_.size());
    points_img_.resize(camera_matrix_.size());

    cv::Mat tracks = cv::Mat::zeros(pp_.y * 2, pp_.x * 2, CV_8UC3);

    for (int cam_idx = 0; cam_idx < tracks_.size(); cam_idx++) {

        for (const auto &track : tracks_[cam_idx]) {

            std::vector<cv::Point2f> points;
            std::vector<cv::Mat_<double>> sfm_points_2d;
            std::vector<cv::Mat_<double>> projection_matrices;

            for (int i = 0; i < track.size(); i++) {
                points.push_back(features_[track[i].first].keypoints[track[i].second].pt);

                LOG(INFO) << track[i].first << features_[track[i].first].keypoints[track[i].second].pt;

                sfm_points_2d.push_back(cv::Mat(points[i]).reshape(1));
                cv::Mat P;
                hconcat(R_[track[i].first], t_[track[i].first], P);

                projection_matrices.push_back(getProjectionMatrix(K_, P));

            }

            LOG(INFO) << "-----";

            if (points.size() < 3) {
                continue;
            }

            cv::Mat_<double> point_3d_mat;
            cv::sfm::triangulatePoints(sfm_points_2d, projection_matrices, point_3d_mat);
            cv::Point3d points3d(point_3d_mat);

            cv::Mat p_origin = R_[cam_idx].t() * (point_3d_mat - t_[cam_idx]);
            double dist = cv::norm(p_origin);

            if (dist < kMax3DDist && p_origin.at<double>(2) > kMin3DDist && std::fabs(p_origin.at<double>(0)) < kMax3DWidth) {

                points_3d_.push_back(points3d);

                std::vector< cv::Point2d > points_img(camera_matrix_.size(), cv::Point2d(0,0));
                std::vector< int > visibility(camera_matrix_.size(), 0);

                for (int i = 0; i < points.size(); i++) {
                    points_img[cam_idx + i] = points[i];
                    visibility[cam_idx + i] = 1;

                    //compute reporjection error for P an

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
    drawViz();
    imshow("tracks", tracks);
    LOG(INFO) << points_3d_.size() << " " << points_img_.size();
}


int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if(points_3d_.size() < 3){
        return 1;
    }

    using namespace gtsam;

    Values result;

    Cal3_S2::shared_ptr K(new Cal3_S2(focal_.x, focal_.y, 0 /* skew */, pp_.x, pp_.y));
    noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

    NonlinearFactorGraph graph;
    Values initial;

    // Poses
    for (size_t pose_idx=0; pose_idx < R_.size(); pose_idx++) {

        Rot3 R(
                R_[pose_idx].at<double>(0,0),
                R_[pose_idx].at<double>(0,1),
                R_[pose_idx].at<double>(0,2),

                R_[pose_idx].at<double>(1,0),
                R_[pose_idx].at<double>(1,1),
                R_[pose_idx].at<double>(1,2),

                R_[pose_idx].at<double>(2,0),
                R_[pose_idx].at<double>(2,1),
                R_[pose_idx].at<double>(2,2)
        );

        Point3 t;

        t(0) = t_[pose_idx].at<double>(0);
        t(1) = t_[pose_idx].at<double>(1);
        t(2) = t_[pose_idx].at<double>(2);

        Pose3 pose(R, t);

        // Add prior for the first image
        if (pose_idx == 0) {
            noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
            graph.push_back(PriorFactor<Pose3>(Symbol('x', pose_idx), pose, pose_noise)); // add directly to graph
        }

        initial.insert(Symbol('x', pose_idx), pose);

        // landmark seen
        for (size_t kp_idx=0; kp_idx < visibility_[pose_idx].size(); kp_idx++) {

            if (visibility_[pose_idx][kp_idx] == 1) {
                Point2 pt;

                pt(0) = points_img_[pose_idx][kp_idx].x;
                pt(1) = points_img_[pose_idx][kp_idx].y;
                graph.push_back(GenericProjectionFactor<Pose3, Point3, Cal3_S2>(pt, measurement_noise, Symbol('x', pose_idx), Symbol('l', kp_idx), K));
            }
        }
    }

    // Add a prior on the calibration.
    //initial.insert(Symbol('K', 0), K);

    //noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 10, 10, 0.01 /*skew*/, 10, 10).finished());
    //graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);

    // Initialize estimate for landmarks
    bool init_prior = false;

    for (size_t kp_idx=0; kp_idx < points_3d_.size(); kp_idx++) {
        initial.insert<Point3>(Symbol('l', kp_idx), Point3(points_3d_[kp_idx].x, points_3d_[kp_idx].y, points_3d_[kp_idx].z));

        if (!init_prior) {
            init_prior = true;

            noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
            Point3 p(points_3d_[kp_idx].x, points_3d_[kp_idx].y, points_3d_[kp_idx].z);
            graph.emplace_shared<PriorFactor<Point3>>(Symbol('l', kp_idx), p, point_noise);
        }
    }

    result = LevenbergMarquardtOptimizer(graph, initial).optimize();

    LOG(INFO) << "initial graph error = " << graph.error(initial);
    LOG(INFO) << "final graph error = " << graph.error(result);

    for (size_t pose_idx=0; pose_idx < R_.size(); pose_idx++) {
        cv::eigen2cv(result.at<Pose3>(Symbol('x', pose_idx)).rotation().matrix(), R_[pose_idx]);
        cv::eigen2cv(result.at<Pose3>(Symbol('x', pose_idx)).translation().vector(), t_[pose_idx]);
    }

    *R = R_[R_.size()-1];
    *t = t_[t_.size()-1];

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

    for (int i=0; i < t_.size(); i++) {

        const cv::Mat & t = t_[i];
        cv::Point2d draw_pos = cv::Point2d(t.at<double>(0) * scale + ba_map.cols / 2, t.at<double>(2) * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(255, 0, 0), 2);

        const cv::Mat &R = R_[i];
        double data[3] = {0, 0, 1};
        cv::Mat dir(3, 1, CV_64FC1, data);

        dir = (R * dir) * 10 * scale;

        cv::line(ba_map, draw_pos, cv::Point2d(dir.at<double>(0, 0), dir.at<double>(0, 2)) + draw_pos,
                 cv::Scalar(0, 255, 255), 2);

    }

    if (!t_.empty()) {
        const auto &t = t_[t_.size() - 1];
        cv::Point2d draw_pos = cv::Point2d(t.at<double>(0) * scale + ba_map.cols / 2, t.at<double>(2) * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("BA_Map", ba_map);
}



void BundleAdjustment::drawViz(){

    int count = 0;
    for (int i = 0; i < t_.size(); i++) {

        const cv::Mat & t = t_[i];
        const cv::Mat & R = R_[i];

        auto col = cv::viz::Color::red();
        if(count % 3 == 1) {
            col = cv::viz::Color::green();
        } else if(count % 3 == 2) {
            col = cv::viz::Color::blue();
        }

        cv::viz::WCameraPosition cam(cv::Matx33d(K_), 3, col);
        viz_.showWidget("c" + std::to_string(count), cam);

        cv::Affine3d pose(R, t);
        viz_.setWidgetPose("c" + std::to_string(count), pose);

        if(count ==0){
            viz_.setViewerPose(pose);
        }
        count++;
    }

    if(!points_3d_.empty()) {
        cv::viz::WCloud cloud_widget(points_3d_, cv::viz::Color::green());

        viz_.showWidget("cloud", cloud_widget);
    }
    viz_.spinOnce(50);
}
