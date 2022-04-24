#ifndef _MTB_H_
#define _MTB_H_
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
Mat image_offset(const Mat& image, int x_offset, int y_offset);
pair<Mat, Mat> to_bitmap(const Mat& image);
vector<Mat> MTB(const vector<Mat>& images, int MTB_iteration = 6, int align_image_index = -1);
#endif