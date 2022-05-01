#include "warping.h"


Mat cylindrical_warping(const Mat& input,vector<FeaturePoint>& feat, double f) {
    Mat dst,mask;
    Mat result(input.rows, input.cols, input.type(), Scalar::all(0));
    mask = Mat(input.rows, input.cols, CV_8UC1, Scalar::all(255));
    int xc = input.cols / 2;
    int yc = input.rows / 2;
    for (int y = 0; y < input.rows; y++)
        for (int x = 0; x < input.cols; x++)
        {
            int x_ = x - xc + 1;
            int y_ = y - yc + 1;
            //cout << "x_: " << x_ << ", y_: " << y_ << endl;
            y_ = y_ * sqrt(1 + pow(tan(x_ / f), 2));
            x_ = f * tan(x_ / f);
            //cout << "x_: " << x_ << ", y_: " << y_ << ", f: " << f << endl;
            x_ += xc - 1;
            y_ += yc - 1;
            if (x_ >= 0.0 && x_ < input.cols && y_ >= 0.0 && y_ < input.rows)
                result.at<Vec3b>(y, x) = input.at<Vec3b>(y_, x_);
            else
            {
                for (int i = -2; i <= 2; i++)
                {
                    if (x + i < 0 || x + i >= input.cols)
                        continue;
                    for (int j = -2; j <= 2; j++)
                    {
                        if (y + j < 0 || y + j >= input.rows)
                            continue;
                        mask.at<uchar>(y + j, x + i) = 0;
                    }
                }
            }
        }
    dst = result;
    for (int index = 0; index < feat.size(); index++)
    {
        int x = feat[index].dx - xc + 1;
        int y = feat[index].dy - yc + 1;
        y = f * y / sqrt(x * x + f * f);
        x = f * atan((float)x / f);
        float at = fastAtan2((float)x, f);
        x += xc - 1;
        y += yc - 1;
        feat[index].dx = x;
        feat[index].dy = y;
    }
    return dst;
}

Mat cylindrical_warping2(const Mat& input, vector<FeaturePoint>& feat) {
    double f = ((int)input.rows / 10) * 10;
    cout << "focal length:"<< f << endl;
    double height = input.rows;
    double width = f * atan2(-(double)input.cols / 2.0, f);
    width = 2 * abs(width);
    //cout << "height = " << height << " , width = " << width << endl;
    Mat result = Mat::zeros(ceil(height), ceil(width),CV_8UC3);

    for (int x = 0; x < result.cols; x++) {
        for (int y = 0; y < result.rows; y++) {
            double tmp_x = x - result.cols / 2.0;
            double tmp_y = y - result.rows / 2.0;
            double source_x = f * tan(tmp_x / f);
            double source_y = tmp_y * sqrt(tmp_x * tmp_x + f * f) / f;
            source_x = source_x + input.cols / 2;
            source_y = source_y + input.rows / 2;
            //
            int l_x = cvFloor(source_x);
            int r_x = cvCeil(source_x);
            int t_y = cvCeil(source_y);
            int d_y = cvFloor(source_y);
            //
            cv::Vec3b ld = cv::Vec3b(0, 0, 0);
            cv::Vec3b lt = cv::Vec3b(0, 0, 0);
            cv::Vec3b rd = cv::Vec3b(0, 0, 0);
            cv::Vec3b rt = cv::Vec3b(0, 0, 0);
            //
            if (l_x >= 0 && l_x < input.cols && d_y >= 0 && d_y < input.rows)
                ld = input.at<Vec3b>(d_y,l_x);
            if (l_x >= 0 && l_x < input.cols && t_y >= 0 && t_y < input.rows)
                lt = input.at<Vec3b>(t_y, l_x);
            if (r_x >= 0 && r_x < input.cols && d_y >= 0 && d_y < input.rows)
                rd = input.at<Vec3b>(d_y, r_x);
            if (r_x >= 0 && r_x < input.cols && t_y >= 0 && t_y < input.rows)
                rt = input.at<Vec3b>(t_y, r_x);
            //
            double a = source_x - l_x;
            double b = source_y - d_y;
            //
            cv::Vec3b t, d, n;
            for (int i = 0; i < 3; i++) {
                if (l_x == r_x) {
                    t.val[i] = lt.val[i];
                    d.val[i] = lt.val[i];
                }
                else {
                    t.val[i] = lt.val[i] * (1 - a) + rt.val[i] * a;
                    d.val[i] = ld.val[i] * (1 - a) + rt.val[i] * a;
                }
            }
            for (int i = 0; i < 3; i++) {
                if (d_y == t_y) {
                    n.val[i] = t.val[i];
                }
                else {
                    n.val[i] = d.val[i] * (1 - b) + t.val[i] * b;
                }
            }
            result.at<Vec3b>(y, x) = n;
        }
    }
    //warp feature point

    for (int i = 0; i < feat.size(); i++) {
        double tmp_x = feat[i].dx - input.cols / 2.0;
        double tmp_y = feat[i].dy - input.rows / 2.0;
        double new_x = f * atan2(tmp_x, f);
        double new_y = f * tmp_y / sqrt(tmp_x * tmp_x + f * f);
        feat[i].dx = new_x + result.cols / 2.0;
        feat[i].dy = new_y + result.rows / 2.0;
    }
    return result;
}