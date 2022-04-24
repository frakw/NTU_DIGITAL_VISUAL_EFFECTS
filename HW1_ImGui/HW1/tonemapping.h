#ifndef _TONEMAPPING_H_
#define _TONEMAPPING_H_
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <iterator>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


pair<Mat, Mat> global_operator(Mat HDR);
pair<Mat, Mat> local_operator(Mat HDR);
pair<Mat, Mat> bilateral_operator(Mat HDR, bool use_cv_bilateral = true);
pair<Mat, Mat> logarithmic_operator(Mat HDR);

#endif