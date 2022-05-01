#ifndef _WARPING_H_
#define _WARPING_H_
#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "feature_point.h"
using namespace std;
using namespace cv;

//Mat cylindrical_warping(const Mat& input, double focal_length);
Mat cylindrical_warping(const Mat& input, vector<FeaturePoint>& feat, double f);
Mat cylindrical_warping2(const Mat& input, vector<FeaturePoint>& feat);

#endif