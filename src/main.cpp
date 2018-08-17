
#include "main.h"

#include <boost/range/iterator_range.hpp>
#include <vector>

#include <cxcore.h>
#include <highgui.h>
#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>

#include "src/features/feature_matcher.h"

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



    cv::Mat output;


    double focal = 707.0912;
    cv::Point2d pp(601.8873, 183.1104);



    bool done = false;

    cv::Mat map(1500, 1500, CV_8UC3, cv::Scalar(0,0,0));

    cv::Mat_<double> pos_t = cv::Mat::zeros(3,1,CV_64FC1);
    cv::Mat_<double> pos_R = cv::Mat::eye(3,3,CV_64FC1);


    FeatureMatcher feature_matcher;

    for (const auto &file_name : file_names) {

        cv::Mat img = cv::imread(file_name);


        feature_matcher.addFrame(img);

        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2f> points0, points1;

        feature_matcher.getMatches(&matches, &points0, &points1);


        if(matches.size() < 8) {
            LOG(WARNING) << "Too few good matches";
            continue;
        }

        cv::Mat E, R, t, mask;

        E = cv::findEssentialMat(points0, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        recoverPose(E, points0, points1, R, t, focal, pp, mask);

        pos_R = R * pos_R;
        pos_t -=  kScale * (pos_R * t);

        cv::Point2d draw_pos = cv::Point2d(pos_t.at<double>(0) + map.cols/2, pos_t.at<double>(2) + map.rows/2);
        cv::circle(map, draw_pos, 2, cv::Scalar(255,0,0), 2);

        cv::imshow("Map", map);
        cv::imshow("Features", feature_matcher.drawMatches());

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27){
            done = true;
            break;
        }
    }


    while(!done) {
        char key = static_cast<char>(cv::waitKey(33));
        if (key == 27) {
            done = true;
        }
    }

}