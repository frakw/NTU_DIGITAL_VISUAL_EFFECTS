#ifndef _SIFT_H_
#define _SIFT_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "feature_point.h"
#include "match_info.h"
using namespace std;
using namespace cv;

vector<FeaturePoint> SIFT(Mat img);

vector<MatchInfo> match(Mat img1, vector<FeaturePoint> fp1, Mat img2, vector<FeaturePoint> fp2);

#endif