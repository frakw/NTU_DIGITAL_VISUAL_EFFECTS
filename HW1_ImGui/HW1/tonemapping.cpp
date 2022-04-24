#include "tonemapping.h"

static float a = 0.36f;
static float l_white = 3.0f;
static float epsilon = 0.05f;
static int gaussian_max_level = 8;
static float gaussian_sigma = 0.35f;
static float gaussian_next_sigma_mult = 1.6f;

pair<Mat, Mat> logarithmic_operator(Mat HDR) {
    Mat intensity(HDR.size(), CV_32FC1);
    double total_brightness = 0.0f;
    double max_luminance = numeric_limits<float>::min();
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            double lw = 0.0722f * HDR.at<Vec3f>(i, j)[0] + 0.7152f * HDR.at<Vec3f>(i, j)[1] + 0.2126f * HDR.at<Vec3f>(i, j)[2]; //計算亮度
            if (isnan(lw)) {
                intensity.at<float>(i, j) = 0.0f;
                continue;
            }
            intensity.at<float>(i, j) = lw;
            total_brightness += log(lw + 0.001f);
            if (lw > max_luminance) max_luminance = lw;
        }
    }
    double lw_average = exp(total_brightness / (double)(HDR.rows * HDR.cols));
    max_luminance = (max_luminance * a) / lw_average;
    Mat result(HDR.size(), CV_32FC3);
    Mat result_8U(HDR.size(), CV_8UC3);
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            float lw = intensity.at<float>(i, j);
            float lm = (a * lw) / lw_average;
            float gamma = 2.2f;
            float p = 4;
            float ld = log10(1.0 + p * lm) / log10(1.0 + p * max_luminance);
            result.at<Vec3f>(i, j)[0] = pow(saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[0] * (ld / lw)) , 1.0f / gamma);
            result_8U.at<Vec3b>(i, j)[0] = pow(saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[0] * (ld * 255.0f / lw)),  1.0f /gamma);
            result.at<Vec3f>(i, j)[1] = pow(saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[1] * (ld / lw)), 1.0f / gamma);
            result_8U.at<Vec3b>(i, j)[1] = pow(saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[1] * (ld * 255.0f / lw)), 1.0f / gamma);
            result.at<Vec3f>(i, j)[2] = pow(saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[2] * (ld / lw)), 1.0f/ gamma);
            result_8U.at<Vec3b>(i, j)[2] = pow(saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[2] * (ld * 255.0f / lw)), 1.0f / gamma) ;
        }
    }
    return make_pair(result, result_8U);
}


pair<Mat,Mat> global_operator(Mat HDR) {
    Mat intensity(HDR.size(), CV_32FC1);
    double total_brightness = 0.0f;
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            double lw = 0.0722f * HDR.at<Vec3f>(i, j)[0] + 0.7152f * HDR.at<Vec3f>(i, j)[1] + 0.2126f * HDR.at<Vec3f>(i, j)[2]; //計算亮度
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
            float lw = intensity.at<float>(i, j);
            float lm = (a * lw) / lw_average;
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


pair<Mat, Mat> local_operator(Mat HDR) {
    Mat intensity(HDR.size(), CV_32FC1);
    double total_brightness = 0.0f;
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            double lw = 0.0722 * HDR.at<Vec3f>(i, j)[0] + 0.7152f * HDR.at<Vec3f>(i, j)[1] + 0.2126f * HDR.at<Vec3f>(i, j)[2]; //計算亮度
            if (isnan(lw)) {
                intensity.at<float>(i, j) = 0.0f;
                continue;
            }
            intensity.at<float>(i, j) = lw;
            total_brightness += log(lw + 0.001f);
        }
    }
    double lw_average = exp(total_brightness / (double)(HDR.rows * HDR.cols));
    Mat lm(HDR.size(),CV_32FC1);
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            float a = 0.36f;
            float lw = intensity.at<float>(i, j);
            lm.at<float>(i,j) = (a * lw) / lw_average;
        }
    }
    vector<Mat> ls1(gaussian_max_level), ls2(gaussian_max_level);
    float sigma = gaussian_sigma, next_sigma = gaussian_sigma * gaussian_next_sigma_mult;
    for (int i = 0; i < gaussian_max_level; i++) {
        cv::GaussianBlur(lm, ls1[i], cv::Size(), sigma, sigma);
        cv::GaussianBlur(lm, ls2[i], cv::Size(), next_sigma, next_sigma);
        sigma *= gaussian_next_sigma_mult;
        next_sigma *= gaussian_next_sigma_mult;
    }
    sigma = gaussian_sigma;
    Mat result(HDR.size(), CV_32FC3);
    Mat result_8U(HDR.size(), CV_8UC3);
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            int fit_level = gaussian_max_level - 1;
            float lw = intensity.at<float>(i, j);
            for (int k = 0; k < gaussian_max_level; k++) {
                float vs = (ls1[k].at<float>(i, j) - ls2[k].at<float>(i, j)) / (std::pow(2.0, gaussian_max_level) * a / (sigma * sigma) + ls1[k].at<float>(i, j));
                if (std::abs(vs) < epsilon) {
                    fit_level = k;
                    break;
                }
                sigma *= gaussian_next_sigma_mult;
            }
            float ld = lm.at<float>(i, j) * (1 + lm.at<float>(i, j) / (l_white* l_white)) / (1 + ls1[fit_level].at<float>(i, j));
        
            result.at<Vec3f>(i, j)[0] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[0] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[0] * (ld * 255.0f / lw));
            result.at<Vec3f>(i, j)[1] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[1] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[1] * (ld * 255.0f / lw));
            result.at<Vec3f>(i, j)[2] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[2] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[2] * (ld * 255.0f / lw));
        }
    }
    return make_pair(result, result_8U);
}

inline unsigned int fast_root(unsigned int x) {
    unsigned int a, b;
    b = x;
    a = x = 0x3f;
    x = b / x;
    a = x = (x + a) >> 1;
    x = b / x;
    a = x = (x + a) >> 1;
    x = b / x;
    x = (x + a) >> 1;
    return(x);
}
#define m_distance(x1,y1,x2,y2) ((float)sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)))
#define m_gaussian(x,sigma) (std::exp(-(x*x)/(2 * (sigma * sigma))) / (2 * CV_PI * (sigma * sigma)))

Mat bilateral_filter(Mat image, int kernel_size, float sigma_i, float sigma_s) {
    Mat result = image.clone();
    if (kernel_size % 2 == 0 || image.type() != CV_32FC1) return result;
    int width = image.cols;
    int height = image.rows;
    int half_kernel_size = (int)kernel_size / 2;
    for (int i = half_kernel_size; i < height - half_kernel_size; i++) {
        for (int j = half_kernel_size; j < width - half_kernel_size; j++) {
            float total = 0.0f;
            double weight_total = 0.0f;
            for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
                for (int g = -half_kernel_size; g <= half_kernel_size; g++) {
                    int row = i + k;
                    int col = j + g;
                    float diff = image.at<float>(row, col) - image.at<float>(i, j);
                    float gi = m_gaussian(diff, sigma_i);
                    float dis = m_distance(i, j, row, col);
                    float gs = m_gaussian(dis, sigma_s);
                    float weight = gi * gs;
                    total += image.at<float>(row, col) * weight;
                    weight_total += weight;
                }
            }
            result.at<float>(i, j) = total / weight_total;
        }
    }
    return result;
}


pair<Mat, Mat> bilateral_operator(Mat HDR,bool use_cv_bilateral) {
    Mat intensity(HDR.size(), CV_32FC1);
    Mat intensity_log(HDR.size(), CV_32FC1);
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            double lw = 0.0722 * HDR.at<Vec3f>(i, j)[0] + 0.7152f * HDR.at<Vec3f>(i, j)[1] + 0.2126f * HDR.at<Vec3f>(i, j)[2]; //計算亮度
            if (isnan(lw)) {
                intensity.at<float>(i, j) = 0.0f;
                continue;
            }
            intensity.at<float>(i, j) = lw;
            intensity_log.at<float>(i, j) = log10(lw);
        }
    }

    Mat base_log;
    if (use_cv_bilateral) {
        bilateralFilter(intensity_log, base_log, -1, 0.4, 0.02 * std::min(HDR.rows, HDR.cols));
    }
    else {
        base_log = bilateral_filter(intensity_log, 5, 0.4f, 0.02f * std::min(HDR.rows, HDR.cols));
    }

    Mat detail_log(HDR.size(), CV_32FC1);
    for (int i = 0; i < detail_log.rows; i++) {
        for (int j = 0; j < detail_log.cols; j++) {
            detail_log.at<float>(i, j) = intensity_log.at<float>(i, j) - base_log.at<float>(i, j);
        }
    }
    double min_log, max_log;
    minMaxLoc(detail_log,&min_log,&max_log);
    double compression_factor = log10(5.0f) / (max_log - min_log);
    double log_absolute_scale = max_log * compression_factor;

    Mat result(HDR.size(), CV_32FC3);
    Mat result_8U(HDR.size(), CV_8UC3);
    for (int i = 0; i < intensity.rows; i++) {
        for (int j = 0; j < intensity.cols; j++) {
            float lw = intensity.at<float>(i, j);
            float ld = pow(10,base_log.at<float>(i, j) * compression_factor + detail_log.at<float>(i, j) - log_absolute_scale);
            result.at<Vec3f>(i, j)[0] = saturate_cast<float>( HDR.at<cv::Vec3f>(i, j)[0] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[0] * (ld * 255.0f / lw));
            result.at<Vec3f>(i, j)[1] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[1] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[1] * (ld * 255.0f / lw));
            result.at<Vec3f>(i, j)[2] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[2] * (ld / lw));
            result_8U.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[2] * (ld * 255.0f / lw));
        }
    }
    return make_pair(result, result_8U);
}