#include "warping.h"
using namespace std;
using namespace cv;


Mat cylindrical_warping(const Mat& input, vector<FeaturePoint>& fps) {
    double focal_length = ((int)input.rows / 10) * 10;
    double height = input.rows;
    double width = focal_length * atan2(-(double)input.cols / 2.0, focal_length);
    width = 2 * abs(width);
    Mat result = Mat::zeros(ceil(height), ceil(width),CV_8UC3);
    for (int col = 0; col < result.cols; col++) {
        for (int row = 0; row < result.rows; row++) {
            double tmpx = col - result.cols / 2.0;
            double tmpy = row - result.rows / 2.0;
            double sourcex = focal_length * tan(tmpx / focal_length);
            double sourcey = tmpy * sqrt(tmpx * tmpx + focal_length * focal_length) / focal_length;
            sourcex = sourcex + input.cols / 2;
            sourcey = sourcey + input.rows / 2;
            int l_col = cvFloor(sourcex);
            int r_col = cvCeil(sourcex);
            int t_row = cvCeil(sourcey);
            int d_row = cvFloor(sourcey);         
            cv::Vec3b ld = cv::Vec3b(0, 0, 0);
            cv::Vec3b lt = cv::Vec3b(0, 0, 0);
            cv::Vec3b rd = cv::Vec3b(0, 0, 0);
            cv::Vec3b rt = cv::Vec3b(0, 0, 0);
            auto legal_pos = [&](const Mat & mat, int _row, int _col)->bool {
                return _col >= 0 && _col < mat.cols&& _row >= 0 && _row < mat.rows;
            };
            if (legal_pos(input, d_row, l_col)) ld = input.at<Vec3b>(d_row, l_col);
            if (legal_pos(input, t_row, l_col)) lt = input.at<Vec3b>(t_row, l_col);
            if (legal_pos(input, d_row, r_col)) rd = input.at<Vec3b>(d_row, r_col);
            if (legal_pos(input, t_row, r_col)) rt = input.at<Vec3b>(t_row, r_col);

            double a = sourcex - l_col;
            double b = sourcey - d_row;
            cv::Vec3b t, d, n;
            for (int i = 0; i < 3; i++) {
                if (l_col == l_col) {
                    t[i] = lt[i];
                    d[i] = lt[i];
                }
                else {
                    t[i] = lt[i] * (1 - a) + rt[i] * a;
                    d[i] = ld[i] * (1 - a) + rt[i] * a;
                }
            }
            for (int i = 0; i < 3; i++) {
                if (d_row == t_row) {
                    n[i] = t[i];
                }
                else {
                    n[i] = d[i] * (1 - b) + t[i] * b;
                }
            }
            result.at<Vec3b>(row, col) = n;
        }
    }
    for (int i = 0; i < fps.size(); i++) {
        double tmpx = fps[i].dx - input.cols / 2.0;
        double tmpy = fps[i].dy - input.rows / 2.0;
        double newx = focal_length * atan2(tmpx, focal_length);
        double newy = focal_length * tmpy / sqrt(tmpx * tmpx + focal_length * focal_length);
        fps[i].dx = newx + result.cols / 2.0;
        fps[i].dy = newy + result.rows / 2.0;
    }
    return result;
}