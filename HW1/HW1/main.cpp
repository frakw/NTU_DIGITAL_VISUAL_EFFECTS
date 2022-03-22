//#define EIGEN_USE_MKL_ALL // Determine if use MKL
//#define EIGEN_VECTORIZE_SSE4_2

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#define _USE_MATH_DEFINES
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include "tinyexpr.h"
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>

#define WINDOW_NAME "HDR"

#define Zfunc(img,COLOR,row,col) ((img).at<Vec3b>((row),(col))[(COLOR)])
//#define Wfunc(val) (val < 128.0f ? val+1.0f:256.0f-val)
#define Wfunc(val) (1.0f - ((float)abs(val-127)/127.0f))


//#define opencv_SVD
//#define eigen_jacobi
#define eigen_SparseQR

#define fixed_sample
using namespace std;
using namespace cv;

void MTB(cv::Mat& inputArray, cv::Mat& outputArray)
{
    int color_count[256] = { 0 };
    int i, j;
    cv::Mat temp;
    if (inputArray.type() == CV_8UC3)
    {
        cv::cvtColor(inputArray, temp, cv::COLOR_BGR2GRAY);
    }
    else
    {
        temp = inputArray.clone();
    }

    //#pragma omp parallel for private(i, j)
    for (j = 0; j < inputArray.rows; j++)
    {
        for (i = 0; i < inputArray.cols; i++)
        {
            color_count[temp.at<uchar>(j, i)]++;
        }
    }

    int threshold_total = 0;
    double thresh;
    for (j = 0; j < inputArray.rows; j++)
    {
        threshold_total += color_count[j];

        if (threshold_total >= (temp.cols * temp.rows) / 2)
        {
            thresh = j;
            break;
        }
    }

    cv::Mat&& dest = cv::Mat::zeros(temp.rows, temp.cols, CV_8UC1);
#pragma omp parallel for private(i, j)
    for (j = 0; j < inputArray.rows; j++)
    {
        for (i = 0; i < inputArray.cols; i++)
        {
            temp.at<uchar>(j, i) > thresh ? dest.at<uchar>(j, i) = 255 : dest.at<uchar>(j, i) = 0;
        }
    }
    outputArray.release();
    outputArray = dest.clone();
    temp.release();
}

void MTBA(std::vector<cv::Mat>& inputArrays, std::vector<cv::Mat>& outputArrays)
{
    cv::Mat sample = inputArrays[0].clone();
    MTB(sample, sample);
    std::vector<int> move_length_x(inputArrays.size(), 0);
    std::vector<int> move_length_y(inputArrays.size(), 0);

    //cut the noise frome the base (first pic)


    for (int j = 1; j < inputArrays.size(); j++)
    {
        cv::Mat temp1 = inputArrays[j].clone();
        MTB(temp1, temp1);
        for (int i = 5; i >= 0; i--)
        {
            cv::Mat temp_samp = sample.clone();
            cv::Mat temp = temp1.clone();

            cv::Mat sampimg = sample.clone();
            cv::Mat otherimg = temp1.clone();

            cv::Mat samp = temp_samp.clone();
            cv::Mat other = temp1.clone();
            cv::resize(samp, samp, cv::Size((int)(sample.cols / pow(2, i + 1)), (int)(sample.rows / pow(2, i + 1))));
            cv::resize(samp, samp, cv::Size((int)(sample.cols / pow(2, i)), (int)(sample.rows / pow(2, i))));
            cv::resize(other, other, cv::Size((int)(sample.cols / pow(2, i + 1)), (int)(sample.rows / pow(2, i + 1))));
            cv::resize(other, other, cv::Size((int)(sample.cols / pow(2, i)), (int)(sample.rows / pow(2, i))));

            //            cv::imshow("samp",samp);
            //            cv::imshow("temp_samp",temp_samp);
                        //cv::imshow("temp",temp);

            for (int b = 0; b < samp.rows; b++)
            {
                for (int a = 0; a < samp.cols; a++)
                {
                    if (samp.at<uchar>(b, a) != sampimg.at<uchar>(b, a))
                    {
                        temp_samp.at<uchar>(b, a) = 0;
                    }
                    if (other.at<uchar>(b, a) != otherimg.at<uchar>(b, a))
                    {
                        temp.at<uchar>(b, a) = 0;
                    }
                    if (sampimg.at<uchar>(b, a) != otherimg.at<uchar>(b, a))
                    {
                        temp.at<uchar>(b, a) = 0;
                    }

                }
            }
            //            cv::imshow("sample",sample);
            //            cv::imshow("samp",samp);
            //            cv::imshow("temp_samp",temp_samp);
            //            cv::imshow("temp",temp);

                        //cv::resize(temp, temp, cv::Size((int)(sample.cols / pow(2, i)), (int)(sample.rows / pow(2, i))));
                        //cv::resize(temp_samp, temp_samp, cv::Size((int)(sample.cols / pow(2, i)), (int)(sample.rows / pow(2, i))));

            double total[9] = { 0 };

            for (int b = 0; b < temp_samp.rows; b++)
            {
                for (int a = 0; a < temp_samp.cols; a++)
                {
                    int count = 0;
                    for (int y = -1; y <= 1; y++)
                    {
                        for (int x = -1; x <= 1; x++)
                        {
                            if (a + move_length_x[j] + x > 0 && a + move_length_x[j] + x < temp.cols && b + move_length_y[j] + y > 0 && b + move_length_y[j] + y < temp.rows)
                            {
                                if ((int)temp_samp.at<uchar>(b, a) != (int)temp.at<uchar>(b + move_length_y[j] + y, a + move_length_x[j] + x))
                                    total[count]++;
                            }
                            count++;
                        }
                    }
                }
            }

            int a = 0;
            //            qDebug () << total[0];
            for (int b = 1; b < 9; b++)
            {
                if (total[a] > total[b])
                {
                    a = b;
                }
                //                qDebug () << total[b];
            }

            //            qDebug() << "=========";
            if (total[a] * 1.5 >= total[4])
            {
                a = 4;
            }

            //            qDebug () << "A:" << a;
            //            qDebug () << "==========";


            switch (a)
            {
            case 0:
                move_length_y[j] = (move_length_y[j] - 1) * 2;
                move_length_x[j] = (move_length_x[j] - 1) * 2;
                break;
            case 1:
                move_length_y[j] = (move_length_y[j] - 1) * 2;
                break;
            case 2:
                move_length_y[j] = (move_length_y[j] - 1) * 2;
                move_length_x[j] = (move_length_x[j] + 1) * 2;
                break;
            case 3:
                move_length_x[j] = (move_length_x[j] - 1) * 2;
                break;
            case 4:
                break;
            case 5:
                move_length_x[j] = (move_length_x[j] + 1) * 2;
                break;
            case 6:
                move_length_y[j] = (move_length_y[j] + 1) * 2;
                move_length_x[j] = (move_length_x[j] - 1) * 2;
                break;
            case 7:
                move_length_y[j] = (move_length_y[j] + 1) * 2;
                break;
            case 8:
                move_length_y[j] = (move_length_y[j] + 1) * 2;
                move_length_x[j] = (move_length_x[j] + 1) * 2;
                break;
            }
            //            qDebug() << move_length_x[j] << " " << move_length_y[j];
            //            qDebug() << "@@@@@@@";
        }


    }

    int minX = move_length_x[0], maxX = move_length_x[0];
    int minY = move_length_y[0], maxY = move_length_y[0];

    for (int i = 1; i < inputArrays.size(); i++)
    {
        if (minX > move_length_x[i]) { minX = move_length_x[i]; }
        if (maxX < move_length_x[i]) { maxX = move_length_x[i]; }
        if (minY > move_length_y[i]) { minY = move_length_y[i]; }
        if (maxY < move_length_y[i]) { maxY = move_length_y[i]; }
    }
    std::vector<cv::Mat> tempdest(inputArrays.size());

    int a, j, i, k;
#pragma parallel for private(a, j, i, k)
    for (a = 0; a < inputArrays.size(); a++)
    {
        cv::Mat&& dest = cv::Mat::zeros(sample.rows + abs(minY) + abs(maxY), sample.cols + abs(minX) + abs(maxX), CV_8UC3);
        for (j = 0; j < inputArrays[a].rows; j++)
        {
            for (i = 0; i < inputArrays[a].cols; i++)
            {
                for (k = 0; k < 3; k++)
                {
                    dest.at<cv::Vec3b>(j + abs(minY) + move_length_y[a], i + abs(minX) + move_length_x[a])[k] = inputArrays[a].at<cv::Vec3b>(j, i)[k];
                }
            }
        }
        tempdest[a] = dest.clone();
    }

    cv::Mat&& dest1 = cv::Mat::zeros(sample.rows + abs(minY) + abs(maxY), sample.cols + abs(minX) + abs(maxX), CV_8UC3);

    int tmp = 0;
    //#pragma parallel for private(a, j, i, k) firstprivate(tmp)
    for (j = 0; j < tempdest[0].rows; j++)
    {
        for (i = 0; i < tempdest[0].cols; i++)
        {
            for (k = 0; k < 3; k++)
            {
                tmp = 0;
                for (a = 0; a < tempdest.size(); a++)
                {
                    tmp += tempdest[a].at<cv::Vec3b>(j, i)[k];
                }

                dest1.at<cv::Vec3b>(j, i)[k] = tmp / tempdest.size();
            }
        }
    }

    outputArrays.clear();
    outputArrays = tempdest;
}


int main() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //std::string image_paths[] = {
    //    "./exposures/img01.jpg","./exposures/img02.jpg","./exposures/img03.jpg","./exposures/img04.jpg","./exposures/img05.jpg","./exposures/img06.jpg",
    //    "./exposures/img07.jpg","./exposures/img08.jpg","./exposures/img09.jpg","./exposures/img10.jpg","./exposures/img11.jpg","./exposures/img12.jpg",
    //    "./exposures/img13.jpg"
    //};
    //float exposure_times[] = {
    //    13.0f,10.0f,4.0f,3.2f,1.0f,0.8f,
    //    1.0f/3.0f,0.25f,1.0f/60.0f,1.0f/80.0f,1.0f/320.0f,1.0f/400.0f,
    //    1.0f/1000.0f
    //};

    //std::string image_paths[] = {
    //"./home/img01.jpg","./home/img02.jpg","./home/img03.jpg","./home/img04.jpg","./home/img05.jpg","./home/img06.jpg",
    //"./home/img07.jpg","./home/img08.jpg","./home/img09.jpg","./home/img10.jpg"
    //};
    //float exposure_times[] = {
    //    1.0f/6.0f,1.0f/10.0f,1.0f/15.0f,1.0f/25.0f,1.0f/40.0f,1.0f/60.0f,
    //    1.0f/100.0f,1.0f/160.0f,1.0f/250.0f,1.0f/400.0f
    //};

    std::string folder = "./home2/";
    int image_count = 11;
    std::string* image_paths = new std::string[image_count];
    float* exposure_times = new float[image_count];
    std::fstream time_file(folder + "time.data");
    for (int i = 1; i <= image_count; i++) {
        std::string path = folder;
        std::string time;
        if (i < 10) path += "img0";
        else path += "img";

        path += std::to_string(i) + ".jpg";
        image_paths[i - 1] = path;
        time_file >> time;
        std::cout << time << '\n';
        exposure_times[i - 1] = te_interp(time.c_str(),0);
        std::cout << exposure_times[i - 1] << '\n';

    }
    //return 0;

    //int image_count = sizeof(exposure_times) / sizeof(exposure_times[0]); // P   
    //Mat* images = new Mat[image_count];
    std::vector<Mat> images;
   
    for (int i = 0; i < image_count; i++) {
        images.push_back(imread(image_paths[i], 1));   
    }
    bool MTB_open = true;
    if (MTB_open) {
        MTBA(images, images);
    }

    Mat* images_sample = new Mat[image_count];
   
    for (int i = 0; i < image_count; i++) {

        images[i] = imread(image_paths[i],1);
#ifdef fixed_sample
        resize(images[i], images_sample[i], Size(12, 16));
#else
        int scale_down = 100;
        resize(images[i], images_sample[i], Size(images[i].cols / scale_down, images[i].rows / scale_down));
#endif // fixed_sample

        
        
        //cv::imshow(std::to_string(i), images[i]);
    }
    int origin_pixel_count = images[0].rows * images[0].cols;

    int pixel_count = images_sample[0].rows * images_sample[0].cols;// N
    int row_count = images_sample[0].rows;
    int col_count = images_sample[0].cols;

    int n = 256;
    float l = 30;
    std::cout << pixel_count * image_count + n + 1 << std::endl;
    std::cout << row_count << ' ' << col_count << std::endl;

#ifdef opencv_SVD
    Mat x[3];
#elif defined(eigen_jacobi)
    Eigen::MatrixXf A(pixel_count * image_count + n + 1, n + pixel_count);
    Eigen::VectorXf b(A.rows());
    Eigen::VectorXf x[3];
#elif defined(eigen_SparseQR)
    Eigen::SparseMatrix<float> A(pixel_count * image_count + n + 1, n + pixel_count);    
    Eigen::VectorXf b(A.rows());
    Eigen::VectorXf x[3];
#endif // !opencv_SVD


#ifdef opencv_SVD
#else
#endif // !opencv_SVD



    
    float Gfunc[3][256];
    float logG[3][256];
    Mat HDR = Mat::zeros(images[0].size(), CV_32FC3);
    Mat HDR_radiance = Mat::zeros(images[0].size(), CV_32FC3);
    //Eigen::VectorXf x(n+ pixel_count);
    auto axes = CvPlot::makePlotAxes();

    bool show_log = true;
    #pragma omp parallel for
    for (int color = 0; color < 3; color++) {
#ifdef opencv_SVD
        Mat A = Mat::zeros(pixel_count * image_count + n + 1, n + pixel_count, CV_32F);
        Mat b = Mat::zeros(A.rows, 1, CV_32F);
#else
        A.setZero();
        b.setZero();
#endif

        int k = 0;
        int i = 0;

        for (int row = 0; row < row_count; row++) {
            for (int col = 0; col < col_count; col++) {
                for (int img_index = 0; img_index < image_count; img_index++) {
                    int z_val = Zfunc(images_sample[img_index], color, row, col);
                    float wij = Wfunc(z_val);

#ifdef opencv_SVD
                    A.at<float>(k, z_val) = wij;
                    A.at<float>(k, n + i) = -wij;
                    b.at<float>(k, 0) = wij * log(exposure_times[img_index]) / log(2.718281828);
#else
                    A.coeffRef(k, z_val) = wij;
                    A.coeffRef(k, n + i) = -wij;
                    b.coeffRef(k) = wij * log(exposure_times[img_index]);
#endif // !opencv_SVD

                    k = k + 1;
                }
                i++;
            }
        }

#ifdef opencv_SVD
        A.at<float>(k, 128) = 1.0f;
#else
        A.coeffRef(k, 128) = 1.0f;
#endif // !opencv_SVD
        
        k = k + 1;
        for (int i = 0; i < n - 2; i++) {
            float lW = l * Wfunc(i+1);

#ifdef opencv_SVD
            A.at<float>(k, i) = lW;
            A.at<float>(k, i +1) = -2 * lW;
            A.at<float>(k, i + 2) = lW;
#else
            A.coeffRef(k, i) = lW;
            A.coeffRef(k, i + 1) = -2 * lW;
            A.coeffRef(k, i + 2) = lW;
#endif // !opencv_SVD

            k = k + 1;
        }



        std::cout << "Ax=b solving..." << std::endl;


        //x[color] = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
#ifdef opencv_SVD
        solve(A, b, x[color], DECOMP_SVD); // Pseudo Inverse
#elif defined(eigen_jacobi)
        x[color] = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
#elif defined(eigen_SparseQR)
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> lscg;
        lscg.compute(A);
        x[color] = lscg.solve(b);
#endif // !opencv_SVD


        std::vector<float> x_index;
        for (int i = 0; i < 256; i++) {
#ifdef opencv_SVD
            logG[color][i] = (x[color].at<float>(i));
            Gfunc[color][i] = exp(x[color].at<float>(i));
#else
            logG[color][i] = (x[color].coeff(i));
            Gfunc[color][i] = exp(x[color].coeff(i));
#endif // !opencv_SVD

            x_index.push_back(i);
        }
        
        
        if (!show_log) {
            switch (color)
            {
            case 0:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-b"); break;
            case 1:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-g"); break;
            case 2:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-r"); break;
            default:
                break;
            }
        }
        else {
            switch (color)
            {
            case 0:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(logG[color]), std::end(logG[color])), "-b"); break;
            case 1:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(logG[color]), std::end(logG[color])), "-g"); break;
            case 2:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(logG[color]), std::end(logG[color])), "-r"); break;
            default:
                break;
            }
        }


    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    //imshow("response curve", axes.render());
    for (int color = 0; color < 3; color++)
    {
        for (int row = 0; row < images[0].rows; row++)
        {
            for (int col = 0; col < images[0].cols; col++)
            {
                float log_sum = 0, weight_sum = 0;

                for (int img_index = 0; img_index < image_count; img_index++)
                {
                    int pixel_val = images[img_index].at<Vec3b>(row, col)(color);

                    log_sum += Wfunc(pixel_val) * log(Gfunc[color][pixel_val] / exposure_times[img_index]);
                    weight_sum += Wfunc(pixel_val);
                }
                float result = exp(log_sum / weight_sum);
                if (isinf(result)) {
                    HDR.at<Vec3f>(row, col)(color) = 0.0f;
                }
                else {
                    HDR.at<Vec3f>(row, col)(color) = result;
                }
                //HDR_radiance.at<Vec3f>(row, col)(color) = log_sum / weight_sum;
                //std::cout << HDR.at<Vec3f>(row, col)(color) << "   " << HDR_radiance.at<Vec3f>(row, col)(color) <<'\n';
                //std::cout << HDR.at<Vec3f>(row, col)(color) << std::endl;
            }
        }
    }
    imshow("HDR", HDR);
   
    cvtColor(HDR, HDR_radiance, COLOR_BGR2GRAY);
    HDR_radiance.convertTo(HDR_radiance,CV_8U,255.0f);
    applyColorMap(HDR_radiance, HDR_radiance, COLORMAP_JET);
    imshow("HDR radiance", HDR_radiance);
    
    imwrite("HDR_Deb_image.exr", HDR);

    //tonemapping/////////

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
    double lw_average = exp(total_brightness / (double)origin_pixel_count);
    Mat tonemapping(HDR.size(), CV_32FC3);
    Mat tonemapping_8U(HDR.size(), CV_8UC3);
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
            tonemapping.at<Vec3f>(i, j)[0] = saturate_cast<float>(HDR.at<cv::Vec3f>(i,j)[0] * (ld / lw));
            tonemapping_8U.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[0] * (ld *255.0f / lw));
            tonemapping.at<Vec3f>(i, j)[1] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[1] * (ld / lw));
            tonemapping_8U.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[1] * (ld * 255.0f / lw));
            tonemapping.at<Vec3f>(i, j)[2] = saturate_cast<float>(HDR.at<cv::Vec3f>(i, j)[2] * (ld / lw));
            tonemapping_8U.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(HDR.at<cv::Vec3f>(i, j)[2] * (ld * 255.0f / lw));
        }
    }
    imshow("After tonemapping", tonemapping);
    imwrite("tonemapping.png", tonemapping_8U);
    std::cout << "finish";
    cv::waitKey();
    return 0;
}