#ifndef ALIGNMENT_
#define ALIGNMENT_

#include <vector>
#include <tuple>

#include <opencv2/opencv.hpp>

std::vector<std::tuple<cv::Mat, double>> alignment(const std::vector<std::tuple<cv::Mat, double>>&, bool);

int get_diff_error(const cv::Mat& center_threshold,
    const cv::Mat& shift_threshold,
    const cv::Mat& center_exclude,
    const cv::Mat& shift_exclude);

std::tuple<cv::Mat, cv::Mat> shift_bitmap(const cv::Mat& threshold,
    const cv::Mat& exclude, int rs,
    int cs);
std::tuple<cv::Mat, cv::Mat> create_bitmap(const cv::Mat& gray);
std::tuple<int, int> get_exp_shift(const cv::Mat& center_image,
    const cv::Mat& image, int shift_bits);

#endif
