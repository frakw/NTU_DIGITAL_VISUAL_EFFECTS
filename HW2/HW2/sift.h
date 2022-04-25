#ifndef _SIFT_H_
#define _SIFT_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include "sift_define.h"
#include "feature_point.h"
#include "match_info.h"
using namespace std;
using namespace cv;

vector<FeaturePoint> SIFT(Mat img);

vector<Mat> get_gaussian_pyramid(Mat img);

vector<Mat> difference_of_gaussian_pyramid(const vector<Mat>& gaussian_pyramid);

vector<MatchInfo> match(Mat img1, vector<FeaturePoint> fp1, Mat img2, vector<FeaturePoint> fp2);

#endif