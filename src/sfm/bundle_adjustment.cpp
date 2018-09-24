
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"
#include <opencv2/sfm/triangulation.hpp>
#include <iostream>

#include "src/utils/draw.h"
#include "src/matcher/matcher.h"


void BundleAdjustment::init(const cv::Mat &K, size_t max_frames) {

    max_frames_ = max_frames;

    K_ = K.clone();

    focal_ = cv::Point2d(K.at<double>(0,0), K.at<double>(1,1));
    pp_ = cv::Point2d(K.at<double>(0,2), K.at<double>(1,2));

    cvsba::Sba::Params param;
    param.fixedIntrinsics = 5;
    param.fixedDistortion = 5;
    sba_.setParams(param);


    viz_ = cv::viz::Viz3d("Coordinate Frame");
    viz_.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

}

//// THIS IS THE PROBLEM http://www.land-of-kain.de/docs/coords/
void BundleAdjustment::addKeyFrame(const VOFrame &frame) {


    camera_matrix_.push_back(K_.clone());
    R_.push_back(frame.pose_R.clone());
    t_.push_back(frame.pose_t.clone());
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
        t_.erase(t_.begin());
        dist_coeffs_.erase(dist_coeffs_.begin());
    }

    pairwise_matches_ = matcher(features_, K_);
    match_matrix_ = createMatchMatrix(pairwise_matches_, R_.size());
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

    for (const auto &row : match_matrix_) {


        std::vector<cv::Point2f> points;
        std::vector<cv::Mat_<double>> sfm_points_2d;
        std::vector<cv::Mat_<double>> projection_matrices;
        std::vector<int> cams;

        for (int cam_idx = 0; cam_idx < row.size(); cam_idx++) {
            if (row[cam_idx] == -1) {
                continue;
            }
            cams.push_back(cam_idx);
            cv::Point2f p2d = features_[cam_idx].keypoints[row[cam_idx]].pt;
            points.push_back(p2d);

            sfm_points_2d.push_back(cv::Mat(p2d).reshape(1));
            cv::Mat P;
            hconcat(R_[cam_idx], t_[cam_idx], P);
            projection_matrices.push_back(getProjectionMatrix(K_, P));
        }

        if (points.size() < 3) {
            continue;
        }


        cv::Mat_<double> point_3d_mat;
        cv::sfm::triangulatePoints(sfm_points_2d, projection_matrices, point_3d_mat); //What is ging on wth this result
        assert(point_3d_mat.type()==CV_64F);
        cv::Point3d points3d(point_3d_mat);

        cv::Mat p_origin = R_[cams[0]].t() * (point_3d_mat - t_[cams[0]]);
        double dist = cv::norm(p_origin);

        if (dist < kMax3DDist && p_origin.at<double>(0,2) > 5) {

            points_3d_.push_back(points3d);

            std::vector< cv::Point2d > points_img(camera_matrix_.size(), cv::Point2d(0,0));
            std::vector< int > visibility(camera_matrix_.size(), 0);

            for (int i = 0; i < points.size(); i++) {
                points_img[cams[i]] = points[i];
                visibility[cams[i]] = 1;

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
    drawViz();
    imshow("tracks", tracks);
    LOG(INFO) << points_3d_.size() << " " << points_img_.size();
}

//TODO use full history rather than just updating the newest point
int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if (points_3d_.empty() || camera_matrix_.size() < 3) {
        LOG(INFO) << "Bundle adjustment points are empty";
        return 1;
    }

    sba_.run(points_3d_, points_img_, visibility_, camera_matrix_, R_, t_, dist_coeffs_);

    LOG(INFO) <<"Initial error="<<sba_.getInitialReprjError()<<". "<<
             "Final error="<<sba_.getFinalReprjError();

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

    for (int i  =0; i < t_.size(); i++) {

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
