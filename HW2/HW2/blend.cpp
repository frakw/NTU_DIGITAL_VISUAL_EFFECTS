#include "blend.h"
using namespace std;
using namespace cv;
void multiBandBlend(cv::Mat& limg, cv::Mat& rimg, int dx, int dy)
{
    if (dx % 2 == 0)
    {
        if (dx + 1 <= limg.cols && dx + 1 <= rimg.cols)
        {
            dx += 1;
        }
        else
        {
            dx -= 1;
        }
    }
    if (dy % 2 == 0)
    {
        if (dy + 1 <= limg.rows && dy + 1 <= rimg.rows)
        {
            dy += 1;
        }
        else
        {
            dy -= 1;
        }
    }


    std::vector<cv::Mat> llpyr, rlpyr;

    buildLaplacianMap(limg, llpyr, dx, dy, LEFT);
    buildLaplacianMap(rimg, rlpyr, dx, dy, RIGHT);

    int center = 0;
    int i, c;
    std::vector<cv::Mat> LS(level);
    for (int a = 0; a < llpyr.size(); a++)
    {
        cv::Mat k = getGaussianKernel(llpyr[a].cols, llpyr[a].rows, llpyr[a].cols);
        LS[a] = cv::Mat(llpyr[a].rows, llpyr[a].cols, CV_32FC3).clone();
        center = (int)(llpyr[a].cols / 2.0);
#pragma omp parallel for private(i, c)
        for (int j = 0; j < LS[a].rows; j++)
        {
            for (i = 0; i < LS[a].cols; i++)
            {
                for (c = 0; c < 3; c++)
                {
                    if (a == llpyr.size() - 1)
                        LS[a].at<cv::Vec3f>(j, i)[c] = llpyr[a].at<cv::Vec3f>(j, i)[c] * k.at<float>(j, i) + rlpyr[a].at<cv::Vec3f>(j, i)[c] * (1.0 - k.at<float>(j, i));
                    else
                        if (i == center)
                        {
                            LS[a].at<cv::Vec3f>(j, i)[c] = (llpyr[a].at<cv::Vec3f>(j, i)[c] + rlpyr[a].at<cv::Vec3f>(j, i)[c]) / 2.0;
                        }
                        else if (i > center)
                        {
                            LS[a].at<cv::Vec3f>(j, i)[c] = rlpyr[a].at<cv::Vec3f>(j, i)[c];
                        }
                        else
                        {
                            LS[a].at<cv::Vec3f>(j, i)[c] = llpyr[a].at<cv::Vec3f>(j, i)[c];
                        }
                }
            }
        }
    }

    cv::Mat result;
    for (int a = level - 1; a > 0; a--)
    {
        cv::pyrUp(LS[a], result, LS[a - 1].size());
#pragma omp parallel for private(i, c)
        for (int j = 0; j < LS[a - 1].rows; j++)
        {
            for (i = 0; i < LS[a - 1].cols; i++)
            {
                for (c = 0; c < 3; c++)
                {
                    LS[a - 1].at<cv::Vec3f>(j, i)[c] = cv::saturate_cast<uchar>(LS[a - 1].at<cv::Vec3f>(j, i)[c] + result.at<cv::Vec3f>(j, i)[c]);
                }
            }
        }
    }

    result = LS[0].clone();

    blendImg(limg, result, dx, dy, LEFT);
    blendImg(rimg, result, dx, dy, RIGHT);
}

cv::Mat getGaussianKernel(int x, int y, int dx, int dy)
{
    cv::Mat kernel = cv::Mat::ones(cv::Size(x, y), CV_32FC1);
    //double sigma = 0.3 * ((dx - 1) * 0.5 -1) + 0.8;
    double half = (dx - 1) / 2.0;

    double sigma = sqrt((-1) * pow((double)kernel.cols - 1 - half, 2.0) / (2 * std::log(0.5)));
    for (int i = (kernel.cols - dx); i < kernel.cols; i++)
    {
        double g;
        if (i <= (kernel.cols - half))
        {
            g = exp((-1) * i * i / (2 * sigma * sigma));
        }
        else
        {
            g = 1 - exp((-1) * pow(kernel.cols - i - 1, 2.0) / (2 * sigma * sigma));
        }

        for (int j = 0; j < kernel.rows; j++)
        {
            kernel.at<float>(j, i) = g;
        }
    }


    //    for(int i = 0; i < kernel.cols; i++)
    //    {
    //        std::cout << kernel.at<float>(0, i) << std::endl;
    //    }
    return kernel;
}

void buildLaplacianMap(cv::Mat& inputArray, std::vector<cv::Mat>& outputArrays, int dx, int dy, int lr)
{

    cv::Mat tmp(cv::Size(dx, abs(dy)), CV_8UC3);


    int disx = (lr == RIGHT) ? 0 : (inputArray.cols - dx);
    int disy = dy >= 0 ? ((lr == RIGHT) ? 0 : (inputArray.rows - dy)) : ((lr == RIGHT) ? (inputArray.rows + dy) : 0);


    if (disx < 0) { disx = 0; }

    for (int j = 0; j < tmp.rows; j++)
    {
        for (int i = 0; i < tmp.cols; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                if (j + disy < inputArray.rows && i + disx < inputArray.cols)
                    tmp.at<cv::Vec3b>(j, i)[c] = inputArray.at<cv::Vec3b>(j + disy, i + disx)[c];
            }
        }
    }
    cv::waitKey();
    tmp.convertTo(tmp, CV_32FC3);

    outputArrays.clear();
    outputArrays.resize(level);

    outputArrays[0] = tmp.clone();
    for (int i = 0; i < level - 1; i++)
    {
        cv::pyrDown(outputArrays[i], outputArrays[i + 1]);
    }

    int i = 0, c = 0;
    for (int a = 0; a < level - 1; a++)
    {
        cv::pyrUp(outputArrays[a + 1], tmp, outputArrays[a].size());

#pragma omp parallel for private(i, c)
        for (int j = 0; j < outputArrays[a].rows; j++)
        {
            for (i = 0; i < outputArrays[a].cols; i++)
            {
                for (c = 0; c < 3; c++)
                {
                    outputArrays[a].at<cv::Vec3f>(j, i)[c] = outputArrays[a].at<cv::Vec3f>(j, i)[c] - tmp.at<cv::Vec3f>(j, i)[c];
                }
            }
        }
    }
}

void blendImg(cv::Mat& inputArray, cv::Mat& overlap_area, int dx, int dy, int lr)
{

    int disx = (lr == RIGHT) ? 0 : (inputArray.cols - dx);
    int disy = dy >= 0 ? ((lr == RIGHT) ? 0 : (inputArray.rows - dy)) : ((lr == RIGHT) ? (inputArray.rows + dy) : 0);

    if (disy < 0) { disy = 0; }
    if (disx < 0) { disx = 0; }

    int  i, c;
#pragma omp parallel for private(i, c)
    for (int j = 0; j < overlap_area.rows; j++)
    {
        for (i = 0; i < overlap_area.cols; i++)
        {
            for (c = 0; c < 3; c++)
            {
                if (j + disy < inputArray.rows && i + disx < inputArray.cols)
                    inputArray.at<cv::Vec3b>(j + disy, i + disx)[c] = cv::saturate_cast<uchar>(overlap_area.at<cv::Vec3f>(j, i)[c]);
            }
        }
    }
}
