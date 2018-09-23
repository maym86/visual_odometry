

#include "utils.h"

// Calculates rotation matrix given euler angles.
cv::Mat eulerAnglesToRotationMatrix(const cv::Mat &theta) {
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
                                 1,       0,              0,
            0,       cos(theta.at<double>(0)),   -sin(theta.at<double>(0)),
            0,       sin(theta.at<double>(0)),   cos(theta.at<double>(0))
    );

    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
                                 cos(theta.at<double>(1)),    0,      sin(theta.at<double>(1)),
            0,               1,      0,
            -sin(theta.at<double>(1)),   0,      cos(theta.at<double>(1))
    );

    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
                                 cos(theta.at<double>(2)),    -sin(theta.at<double>(2)),      0,
            sin(theta.at<double>(2)),    cos(theta.at<double>(2)),       0,
            0,               0,                  1);


    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;

    return R;

}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Mat rotationMatrixToEulerAngles(const cv::Mat &R) {


    cv::Mat euler(3, 1, CV_64F);

    double m00 = R.at<double>(0, 0);
    double m02 = R.at<double>(0, 2);
    double m10 = R.at<double>(1, 0);
    double m11 = R.at<double>(1, 1);
    double m12 = R.at<double>(1, 2);
    double m20 = R.at<double>(2, 0);
    double m22 = R.at<double>(2, 2);

    double x, y, z;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        x = 0;
        y = CV_PI / 2;
        z = atan2(m02, m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        x = 0;
        y = -CV_PI / 2;
        z = atan2(m02, m22);
    }
    else
    {
        x = atan2(-m12, m11);
        y = asin(m10);
        z = atan2(-m20, m00);
    }

    euler.at<double>(0) = x;
    euler.at<double>(1) = y;
    euler.at<double>(2) = z;

    return euler;
}


