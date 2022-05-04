#ifndef _WARPING_H_
#define _WARPING_H_
#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "feature_point.h"


cv::Mat cylindrical_warping(const cv::Mat& input, std::vector<FeaturePoint>& feat);

#endif