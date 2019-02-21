
#include "matcher.h"
#include <cv.hpp>
#include <opencv2/video/tracking.hpp>

const float kMatchRatio = 0.7;

std::vector<cv::detail::MatchesInfo> matcher(const std::vector<cv::detail::ImageFeatures> &features, const cv::Mat &K) {

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::detail::MatchesInfo> pairwise_matches;

    for (int i = 0; i < features.size() - 1; i++) {
        for (int j = i+1; j < std::min(static_cast<int>(features.size()),i+3) ; j++) {

            std::vector<std::vector<cv::DMatch>> matches;
            matcher->knnMatch(features[i].descriptors, features[j].descriptors, matches, 2); // Find two nearest matches

            cv::detail::MatchesInfo good_matches;

            std::vector<cv::Point2f> points0;
            std::vector<cv::Point2f> points1;
            for (int k = 0; k < matches.size(); k++) {

                if (matches[k][0].distance < kMatchRatio * matches[k][1].distance) {

                    const auto &p0 = features[i].keypoints[matches[k][0].queryIdx];
                    const auto &p1 = features[j].keypoints[matches[k][0].trainIdx];

                    if (cv::norm(cv::Mat(p0.pt) - cv::Mat(p1.pt)) < 200) {
                        good_matches.matches.push_back(matches[k][0]);
                        good_matches.inliers_mask.push_back(1);

                        points0.push_back(p0.pt);
                        points1.push_back(p1.pt);
                    }
                }
            }
            cv::Mat mask, R, t;

            if (points0.size() >= 5) {
                cv::Mat E = cv::findEssentialMat(points0, points1, K, cv::RANSAC, 0.999, 1.0, mask);

                good_matches.inliers_mask = mask;

                good_matches.src_img_idx = i;
                good_matches.dst_img_idx = j;
                pairwise_matches.push_back(good_matches);
            }
        }
    }
    return pairwise_matches;
}

std::vector<std::vector<int>> createMatchMatrix(const std::vector<cv::detail::MatchesInfo> pairwise_matches, const int pose_count) {

    std::vector<std::vector<int>> match_matrix;

    for (auto &pwm : pairwise_matches) {
        for (int i = 0; i < pwm.matches.size(); i++) {

            if (pwm.inliers_mask[i] != 1) {
                continue;
            }

            auto &match = pwm.matches[i];
            bool found = false;
            for(auto &row : match_matrix){
                if(row[pwm.src_img_idx] == match.queryIdx || row[pwm.dst_img_idx] == match.trainIdx) {
                    row[pwm.src_img_idx] = match.queryIdx;
                    row[pwm.dst_img_idx] = match.trainIdx;
                    found = true;
                    break;
                }
            }

            if(!found){
                std::vector<int> row(pose_count, -1);
                row[pwm.src_img_idx] = match.queryIdx;
                row[pwm.dst_img_idx] = match.trainIdx;
                match_matrix.push_back(std::move(row));
            }
        }
    }
    return match_matrix;
}


//Experimental test code
std::vector<cv::detail::MatchesInfo> kltBacktrack(const std::vector<cv::Mat> &images, const cv::Mat &K, std::vector<cv::detail::ImageFeatures> *features) {

    cv::Ptr<cv::SparsePyrLKOpticalFlow> optical_flow = cv::SparsePyrLKOpticalFlow::create();

    std::vector<cv::detail::MatchesInfo> pairwise_matches;

    std::vector<cv::Point2f> points_current;
    for(const auto &kp : (*features)[features->size()-1].keypoints){
        points_current.push_back(kp.pt);
    }

    std::vector<cv::Point2f> points_previous;

    std::vector<unsigned char> status;

    for (int i = images.size()-1; i > 0; i--) {
        optical_flow->calc(images[i], images[i-1], points_current, points_previous, status);

        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(points_current, points_previous, K, cv::RANSAC, 0.999, 1.0, mask);
        cv::detail::MatchesInfo good_matches;
        //Remove bad points

        int count = 0;
        for (int i =0; i < status.size(); i++) {
            cv::Point2f pt = points_current[i];
            if(mask.at<uchar>(i) == 0){
                points_previous.erase(points_previous.begin() + i - count);
                count++;
            } else if ((status)[i] == 0 || pt.x < 0 || pt.y < 0) {
                if (pt.x < 0 || pt.y < 0) {
                    (status)[i] = 0;
                }
                points_previous.erase(points_previous.begin() + i - count);
                count++;
            }else{
                //good
                cv::DMatch match;
                match.queryIdx = i;
                match.trainIdx = i - count;
                good_matches.matches.push_back(match);
            }
        }

        (*features)[i-1].keypoints.clear();
        for(const auto &p: points_previous){
            cv::KeyPoint kp;
            kp.pt = p;
            (*features)[i-1].keypoints.push_back(kp);
        }

        good_matches.inliers_mask.resize(points_previous.size(),1);
        good_matches.src_img_idx = i;
        good_matches.dst_img_idx = i-1;
        pairwise_matches.push_back(good_matches);
        points_current = points_previous;
    }
    return pairwise_matches;
}