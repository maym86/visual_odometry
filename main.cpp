
#include "main.h"

#include <boost/range/iterator_range.hpp>
#include <vector>
using std::vector;

#include <cxcore.h>
#include <highgui.h>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>

using namespace boost::filesystem;

int main(int argc, char *argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    //Iterate through directory
    path p(FLAGS_dir);

    if (!is_directory(p)) {
        LOG(INFO) << FLAGS_dir + " is not a directory";
        return 0;
    }

    std::vector<std::string> file_names;
    for (auto &entry : boost::make_iterator_range(directory_iterator(p), {})) {
        file_names.push_back(entry.path().string());
    }
    std::sort(file_names.begin(), file_names.end());

    std::list<cv::Mat> images;
    std::list<std::vector<cv::KeyPoint>> keypoints;
    std::list<cv::Mat> descriptors;

    cv::Mat output;


    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(2000);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    cv::FlannBasedMatcher matcher;

    bool first = true;

    cv::Mat map(1000, 1000, CV_8UC3, cv::Scalar(0,0,0));

    cv::Mat_<double> pos_t = cv::Mat::zeros(3,1,CV_64FC1);
    cv::Mat_<double> pos_R = cv::Mat::eye(3,3,CV_64FC1);

    for (const auto &file_name : file_names) {

        cv::Mat img = cv::imread(file_name);

        std::vector<cv::KeyPoint> kp;
        detector->detect(img, kp);

        cv::Mat desc;
        extractor->compute(img, kp, desc);

        images.push_back(img);
        keypoints.push_back(std::move(kp));

        desc.convertTo(desc, CV_32F);

        descriptors.push_back(desc);

        if(images.size() < 2){
            continue;
        }

        if(images.size() > 2){
            images.pop_front();
            keypoints.pop_front();
            descriptors.pop_front();
        }

        std::vector<cv::DMatch> matches;

        matcher.match(descriptors.front(), descriptors.back(), matches);
        std::vector<cv::DMatch> good_matches;

        const auto &kp0 = keypoints.front();
        const auto &kp1 = keypoints.back();

        std::vector<cv::Point2f> selected_points0, selected_points1;

        for( int i = 0; i < matches.size(); i++ ) {
            if( matches[i].distance <= kMinDist) { //TODO revisit use Lowes method?? 0.8
                good_matches.push_back( matches[i]);
                selected_points0.push_back(kp0[matches[i].queryIdx].pt);
                selected_points1.push_back(kp1[matches[i].trainIdx].pt);
            }
        }

        double focal = 1.0;
        cv::Point2d pp(0.0, 0.0);
        cv::Mat E, R, t, mask;

        E = cv::findEssentialMat(selected_points0, selected_points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        recoverPose(E, selected_points0, selected_points1, R, t, focal, pp, mask);

        ///TODO get this right
        pos_R = R * pos_R;
        pos_t = pos_t +  (pos_R * t);

        cv::Point2d draw_pos = cv::Point2d(pos_t.at<double>(0) *kScale + 500, pos_t.at<double>(2) *kScale + 500);
        cv::circle(map, draw_pos, 2, cv::Scalar(255,0,0), 2);

        drawMatches(images.front(), keypoints.front(), images.back(), keypoints.back(), good_matches, output);

        cv::imshow("Map", map);
        cv::imshow("Features", output);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27){
            break;
        }
    }


}