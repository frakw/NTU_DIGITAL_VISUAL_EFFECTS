#include "MTB.h"

Mat image_offset(const Mat& image, int x_offset, int y_offset) {
    Mat result = Mat::zeros(image.size(), image.type());
    auto valid_xy = [&](int row, int col)->bool {
        return (row >= 0) && (row < image.rows) && (col >= 0) && (col < image.cols);
    };
    if (image.type() == CV_8UC1) {
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int src_row = i - y_offset;
                int src_col = j - x_offset;
                if (valid_xy(src_row, src_col)) {
                    result.at<uchar>(i, j) = image.at<uchar>(src_row, src_col);
                }
            }
        }
    }
    else if (image.type() == CV_8UC3) {
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int src_row = i - y_offset;
                int src_col = j - x_offset;
                if (valid_xy(src_row, src_col)) {
                    result.at<Vec3b>(i, j) = image.at<Vec3b>(src_row, src_col);
                }
            }
        }
    }

    return result;
}

pair<Mat, Mat> to_bitmap(const Mat& image) {
    Mat result(image.size(), CV_8UC1);
    Mat exclude(image.size(), CV_8UC1);
    int range = 4;

    vector<uchar> pixel_values;
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            pixel_values.push_back(image.at<uchar>(i, j));
        }
    }
    sort(pixel_values.begin(), pixel_values.end());
    uchar median = pixel_values[pixel_values.size() / 2];
    int up_bound = median + range;
    int down_bound = median - range;
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.at<uchar>(i, j) = image.at<uchar>(i, j) > median ? 255 : 0;
            exclude.at<uchar>(i, j) = image.at<uchar>(i, j) > up_bound || image.at<uchar>(i, j) < down_bound ? 255 : 0;

        }
    }
    return make_pair(result, exclude);
}

vector<Mat> MTB(const vector<Mat>& images, int MTB_iteration , int align_image_index) {
    if (align_image_index == -1) {
        align_image_index = images.size() / 2;
    }
    Mat align_standard = images[align_image_index];
    int rows = align_standard.rows;
    int cols = align_standard.cols;
    vector<Mat> result(images.size());
    Mat* standard_pyramid = new Mat[MTB_iteration];
    Mat* target_pyramid = new Mat[MTB_iteration];
    cvtColor(align_standard, standard_pyramid[0], cv::COLOR_BGR2GRAY);
    for (int i = 1; i < MTB_iteration; i++) { //產生金字塔，縮小要由上一層當來源，不能直接拿原圖縮
        resize(standard_pyramid[i - 1], standard_pyramid[i], cv::Size(), 0.5, 0.5,
            cv::INTER_NEAREST);
    }
    for (int i = 0; i < images.size(); i++) {
        if (i == align_image_index) {
            result[i] = align_standard;
            continue;
        }
        cvtColor(images[i], target_pyramid[0], cv::COLOR_BGR2GRAY);
        for (int j = 1; j < MTB_iteration; j++) { //產生金字塔，縮小要由上一層當來源，不能直接拿原圖縮
            cv::resize(target_pyramid[j - 1], target_pyramid[j], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        }
        int x_offset = 0;
        int y_offset = 0;
        for (int j = 5; j >= 0; j--) {
            pair<Mat, Mat> standard_bitmap = to_bitmap(standard_pyramid[j]);
            pair<Mat, Mat> target_bitmap = to_bitmap(target_pyramid[j]);
            int min_diff = numeric_limits<int>::max();
            int min_x_offset = x_offset, min_y_offset = y_offset;
            for (int k = -1; k <= 1; k++) {
                for (int g = -1; g <= 1; g++) {
                    int neighbor_x_offset = x_offset + g;
                    int neighbor_y_offset = y_offset + k;
                    Mat target_shift, target_exclude;
                    target_shift = image_offset(target_bitmap.first, neighbor_x_offset, neighbor_y_offset);
                    target_exclude = image_offset(target_bitmap.second, neighbor_x_offset, neighbor_y_offset);
                    Mat diff(standard_bitmap.first.size(), CV_8U);
                    bitwise_xor(standard_bitmap.first, target_shift, diff);
                    bitwise_and(diff, standard_bitmap.second, diff);
                    bitwise_and(diff, target_exclude, diff);
                    int diff_count = countNonZero(diff);
                    if (min_diff > diff_count) {
                        min_x_offset = neighbor_x_offset;
                        min_y_offset = neighbor_y_offset;
                        min_diff = diff_count;
                    }
                }
            }
            x_offset = min_x_offset;
            y_offset = min_y_offset;
            if (j != 0) {
                x_offset *= 2;
                y_offset *= 2;
            }
        }
        result[i] = image_offset(images[i], x_offset, y_offset);
    }
    delete[] standard_pyramid;
    delete[] target_pyramid;
    return result;
}