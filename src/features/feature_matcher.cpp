#include "feature_matcher.h"


FeatureMatcher::FeatureMatcher(){

    detector_ = cv::ORB::create(1000);
    extractor_ = cv::ORB::create();
}

void FeatureMatcher::addFrame(cv::Mat image) {

    std::vector<cv::KeyPoint> kp;
    detector_->detect(image, kp);

    cv::Mat desc;
    extractor_->compute(image, kp, desc);

    images_.push_back(image);
    keypoints_.push_back(std::move(kp));

    desc.convertTo(desc, CV_32F);

    descriptors_.push_back(desc);

    if(images_.size() > 2){
        images_.pop_front();
        keypoints_.pop_front();
        descriptors_.pop_front();
    }
}


void FeatureMatcher::getMatches(std::vector<cv::DMatch> *good_matches, std::vector<cv::Point2f> *good_points0, std::vector<cv::Point2f> *good_points1){

    if(descriptors_.size() != 2){
        return;
    }

    std::vector<cv::DMatch> initial_matches;

    matcher_.match(descriptors_.front(), descriptors_.back(), initial_matches);

    const auto &kp0 = keypoints_.front();
    const auto &kp1 = keypoints_.back();

    for( int i = 0; i < initial_matches.size(); i++ ) {
        if( initial_matches[i].distance <= kMinDist) { //TODO revisit use Lowes method?? 0.8
            good_matches->push_back( initial_matches[i]);
            good_points0->push_back(kp0[initial_matches[i].queryIdx].pt);
            good_points1->push_back(kp1[initial_matches[i].trainIdx].pt);
        }
    }

    matches_ = *good_matches;
}


cv::Mat FeatureMatcher::drawMatches() {

    cv::Mat output;
    if(images_.size() == 2){
        cv::drawMatches(images_.front(), keypoints_.front(), images_.back(), keypoints_.back(), matches_, output);
    }

    return output;
}