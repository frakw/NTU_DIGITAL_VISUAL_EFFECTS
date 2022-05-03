#ifndef _BLEND_H_
#define _BLEND_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#define RIGHT 1
#define LEFT 0

void multiBandBlend(cv::Mat& limg, cv::Mat& rimg, int dx, int dy);
cv::Mat getGaussianKernel(int x, int y, int dx, int dy = 0);
void buildLaplacianMap(cv::Mat& inputArray, std::vector<cv::Mat>& outputArrays, int dx, int dy, int lr);
void blendImg(cv::Mat& img, cv::Mat& overlap_area, int dx, int dy, int lr);
const int level = 5;
#endif