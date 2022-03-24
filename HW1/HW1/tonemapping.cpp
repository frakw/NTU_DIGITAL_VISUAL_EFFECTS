#include "tonemapping.h"

pair<Mat,Mat> global_operator(Mat HDR) {
    Mat intensity(HDR.size(), CV_32FC1);
    double total_brightness = 0.0f;
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            double lw = 0.0722 * HDR.at<Vec3f>(i, j)[0] + 0.7152f * HDR.at<Vec3f>(i, j)[1] + 0.2126f * HDR.at<Vec3f>(i, j)[2]; //­pºâ«G«×
            if (isnan(lw)) {
                intensity.at<float>(i, j) = 0.0f;
                continue;
            }
            intensity.at<float>(i, j) = lw;
            total_brightness += log(lw + 0.001f);
        }
    }
    double lw_average = exp(total_brightness / (double)(HDR.rows * HDR.cols));
    Mat result(HDR.size(), CV_32FC3);
    Mat result_8U(HDR.size(), CV_8UC3);
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            float a = 0.36f;
            float lw = intensity.at<float>(i, j);
            float lm = (a * lw) / lw_average;
            float l_white = 3.0f;
            float ld = 0.0f;
            if (lm >= l_white) {
                ld = 1.0f;
            }
            else {
                ld = (lm * (1.0f + lm / (l_white * l_white))) / (1.0f + lm);
            }
            result.at<Vec3f>(i, j)[0] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[0] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[0] * (ld * 255.0f / lw));
            result.at<Vec3f>(i, j)[1] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[1] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[1] * (ld * 255.0f / lw));
            result.at<Vec3f>(i, j)[2] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[2] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[2] * (ld * 255.0f / lw));
        }
    }
    return make_pair(result,result_8U);
}