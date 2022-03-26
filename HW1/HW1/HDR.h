#ifndef _HDR_H_
#define _HDR_H_
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


#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>

using namespace std;
using namespace cv;

Mat Debevec_HDR_recover(vector<Mat> images, vector<float> exposure_times);
Mat Robertson_HDR_recover(vector<Mat> images, vector<float> exposure_times, int iteration = 30);

#endif