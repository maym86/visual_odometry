#include "feature_matcher.h"


FeatureMatcher::FeatureMatcher(){

    detector_ = cv::ORB::create(kMinFeatures);
    extractor_ = cv::ORB::create();
}

void FeatureMatcher::addFrame(const cv::Mat &image) {

    images_.push_back(image);

    std::vector<cv::KeyPoint> kp;
    detector_->detect(image, kp);

    cv::Mat desc;
    extractor_->compute(image, kp, desc);


    keypoints_.push_back(std::move(kp));

    desc.convertTo(desc, CV_32F);

    descriptors_.push_back(desc);

    if(images_.size() > 2){
        images_.pop_front();
        keypoints_.pop_front();
        descriptors_.pop_front();
    }
}


void FeatureMatcher::getMatches(std::vector<cv::DMatch> *matches, std::vector<cv::Point2f> *points0, std::vector<cv::Point2f> *points1){

    if(descriptors_.size() != 2){
        return;
    }

    std::vector<std::vector<cv::DMatch>> initial_matches;

    matcher_.knnMatch(descriptors_.front(), descriptors_.back(), initial_matches, 2);

    const auto &kp0 = keypoints_.front();
    const auto &kp1 = keypoints_.back();


    for (int i = 0; i < initial_matches.size(); ++i)
    {
        if (initial_matches[i][0].distance < kMatchRatio * initial_matches[i][1].distance)
        {
            matches->push_back( initial_matches[i][0]);
            points0->push_back(kp0[initial_matches[i][0].queryIdx].pt);
            points1->push_back(kp1[initial_matches[i][0].trainIdx].pt);
        }
    }

    matches_ = *matches;
}

