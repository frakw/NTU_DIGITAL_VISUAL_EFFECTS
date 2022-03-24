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

#endif