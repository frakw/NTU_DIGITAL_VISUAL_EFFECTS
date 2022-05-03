#ifndef _WARPING_H_
#define _WARPING_H_
#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "feature_point.h"

//Mat cylindrical_warping(const Mat& input, double focal_length);
cv::Mat cylindrical_warping(const cv::Mat& input, std::vector<FeaturePoint>& feat, double f);
cv::Mat cylindrical_warping2(const cv::Mat& input, std::vector<FeaturePoint>& feat);

#endif