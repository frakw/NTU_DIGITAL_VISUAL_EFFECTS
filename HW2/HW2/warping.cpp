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