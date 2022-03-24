//#define EIGEN_USE_MKL_ALL // Determine if use MKL
//#define EIGEN_VECTORIZE_SSE4_2

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <iterator>

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
#include "alignment.h"
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

const double a = 0.18;
const double eps = 0.05;
const double phi = 8.0;

cv::Mat global_operator(const cv::Mat& radiance_map, const cv::Mat& Lm,
    const double Lwhite) {
    auto rows = radiance_map.rows;
    auto cols = radiance_map.cols;
    cv::Mat Ld(rows, cols, CV_64FC1);

    for (int i = 0; i != rows; i++) {
        for (int j = 0; j != cols; j++) {
            double Lm_l = Lm.at<double>(i, j);
            Ld.at<double>(i, j) = (Lm_l * (1 + Lm_l / std::pow(Lwhite, 2))) / (1 + Lm_l);
        }
    }

    return Ld;
}

cv::Mat local_operator(const cv::Mat& radiance_map, const cv::Mat& Lm,
    const double Lwhite) {
    auto rows = radiance_map.rows;
    auto cols = radiance_map.cols;
    cv::Mat Ld(rows, cols, CV_64FC1);

    const double alpha_1 = 0.35, alpha_2 = 0.35 * 1.6;
    double s = 1.0;
    std::vector<cv::Mat> v1s, v2s;
    for (int i = 0; i != 8; i++) {
        cv::Mat v1(rows, cols, CV_64FC1);
        cv::Mat v2(rows, cols, CV_64FC1);
        cv::GaussianBlur(Lm, v1, cv::Size(), alpha_1 * s, alpha_1 * s,
            cv::BORDER_REPLICATE);
        cv::GaussianBlur(Lm, v2, cv::Size(), alpha_2 * s, alpha_2 * s,
            cv::BORDER_REPLICATE);
        v1s.push_back(v1);
        v2s.push_back(v2);

        s *= 1.6;
    }

    s = 1.0;
    for (int i = 0; i != rows; i++) {
        for (int j = 0; j != cols; j++) {
            int smax = 7;
            for (int k = 0; k != 8; k++) {
                auto v = (v1s[k].at<double>(i, j) - v2s[k].at<double>(i, j)) /
                    (std::pow(2.0, phi) * a / (s * s) + v1s[k].at<double>(i, j));
                if (std::abs(v) < eps) {
                    smax = k;
                    break;
                }

                s *= 1.6;
            }

            auto Lm_l = Lm.at<double>(i, j);
            Ld.at<double>(i, j) = Lm_l * (1 + Lm_l / std::pow(Lwhite, 2)) /
                (1 + v1s[smax].at<double>(i, j));
        }
    }

    return Ld;
}

cv::Mat tone_mapping(const cv::Mat& radiance_map, const int tone = 2) {
    std::cout << "[Tone mapping...]" << std::endl;

    if (tone == 3)
        //return contrast(radiance_map);

    if (tone == 0)
        std::cout << "\tblend global and local operator" << std::endl;
    else if (tone == 1)
        std::cout << "\tglobal operator" << std::endl;
    else
        std::cout << "\tlocal operator" << std::endl;

    auto rows = radiance_map.rows;
    auto cols = radiance_map.cols;

    cv::Mat Lw(rows, cols, CV_64FC1);
    double lum_mean = 0.0;
    double Lwhite = 0.0;

    for (int i = 0; i != rows; i++) {
        for (int j = 0; j != cols; j++) {
            auto value = radiance_map.at<cv::Vec3d>(i, j);
            auto lum = Lw.at<double>(i, j) =
                0.27 * value[0] + 0.67 * value[1] + 0.06 * value[2];
            for (int c = 0; c != 3; c++) Lwhite = std::max(Lwhite, value[c]);
            lum_mean += std::log(0.000001 + lum);
        }
    }
    lum_mean = std::exp(lum_mean / (rows * cols));
    std::cout << "\tLwhite: " << Lwhite << std::endl;

    cv::Mat Lm(rows, cols, CV_64FC1);
    for (int i = 0; i != rows; i++)
        for (int j = 0; j != cols; j++)
            Lm.at<double>(i, j) = Lw.at<double>(i, j) * a / lum_mean;

    cv::Mat tonemap(rows, cols, CV_64FC3);
    double blend = 0.5;
    if (tone == 1) blend = 1.0;
    if (tone == 2) blend = 0.0;
    auto Ld_g = tone != 2 ? global_operator(radiance_map, Lm, Lwhite)
        : cv::Mat::zeros(radiance_map.size(), CV_64FC1);
    auto Ld_l = tone != 1 ? local_operator(radiance_map, Lm, Lwhite)
        : cv::Mat::zeros(radiance_map.size(), CV_64FC1);
    for (int i = 0; i != rows; i++)
        for (int j = 0; j != cols; j++)
            for (int channel = 0; channel != 3; channel++) {
                auto Ld = Ld_g.at<double>(i, j) * (blend)+
                    Ld_l.at<double>(i, j) * (1.0 - blend);
                auto value = radiance_map.at<cv::Vec3d>(i, j)[channel] * Ld /
                    Lw.at<double>(i, j);
                // value = std::pow(value, 1 / 1.2);
                tonemap.at<cv::Vec3d>(i, j)[channel] = value * 255;
            }

    return tonemap;
}


pair<Mat,Mat> to_bitmap(const Mat& image) {
    Mat result(image.size(), CV_8UC1);
    Mat exclude(image.size(),CV_8UC1);
    int range = 4;

    vector<uchar> pixel_values;
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            pixel_values.push_back(image.at<uchar>(i, j));
        }
    }
    sort(pixel_values.begin(), pixel_values.end());
    uchar median = pixel_values[pixel_values.size()/2];
    int up_bound = median + range;
    int down_bound = median - range;
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.at<uchar>(i, j) = image.at<uchar>(i, j) > median ? 255 : 0;
            exclude.at<uchar>(i, j) = image.at<uchar>(i, j) > up_bound || image.at<uchar>(i, j) < down_bound ? 255 : 0;

        }
    }
    return make_pair(result,exclude);
}

Mat to_bitmap2(const Mat& image) {
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
            if (image.at<uchar>(i, j) > up_bound) {
                result.at<uchar>(i, j) = 1;
            }
            else if (image.at<uchar>(i, j) < down_bound) {
                result.at<uchar>(i, j) = 0;
            }
            else {
                result.at<uchar>(i, j) = 2;
            }

        }
    }
    return result;
}

vector<Mat> MTB(const vector<Mat>& images,int MTB_iteration = 6, int align_image_index = -1) {
    if (align_image_index == -1) {
        align_image_index = images.size() / 2;
    }
    Mat align_standard = images[align_image_index];
    int rows = align_standard.rows;
    int cols = align_standard.cols;
    vector<Mat> result(images.size());
    int dir[9][2] = {
        {-1,-1},{0,-1},{1,-1},{-1,0},{0,0},{1,0},{-1,1},{0,1},{1,1},
    };
   
    Mat* standard_pyramid = new Mat[MTB_iteration];
    Mat* target_pyramid = new Mat[MTB_iteration];
    cvtColor(align_standard, standard_pyramid[0], cv::COLOR_BGR2GRAY);
    for (int i = 1; i < MTB_iteration; i++) { //產生金字塔，縮小要由上一層當來源，不能直接拿原圖縮
        cv::resize(standard_pyramid[i - 1], standard_pyramid[i], cv::Size(), 0.5, 0.5,
            cv::INTER_NEAREST);
    }


    for (int i = 0; i < images.size(); i++) {
        if (i == align_image_index) {
            result[i] = align_standard;
            continue;
        }        
        cv::cvtColor(images[i], target_pyramid[0], cv::COLOR_BGR2GRAY);
        for (int j = 1; j < MTB_iteration; j++) { //產生金字塔，縮小要由上一層當來源，不能直接拿原圖縮
            cv::resize(target_pyramid[j - 1], target_pyramid[j], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        }
        int x_offset = 0;
        int y_offset = 0;
        for (int j = 5; j >=0; j--) {
            cout << standard_pyramid[j].size() << " ";
            pair<Mat, Mat> standard_bitmap = to_bitmap(standard_pyramid[j]);
            pair<Mat, Mat> target_bitmap = to_bitmap(target_pyramid[j]);
            //Mat standard_bitmap = to_bitmap2(standard_pyramid[j]);
            //Mat target_bitmap = to_bitmap2(target_pyramid[j]);
            
            int min_diff = numeric_limits<int>::max();
            int min_x_offset = x_offset, min_y_offset = y_offset;
            for (int k = -1; k <= 1; k++) {
                for (int g = -1; g <= 1; g++) {
                    int neighbor_x_offset = x_offset + g;
                    int neighbor_y_offset = y_offset + k;
                    Mat shift_threshold, shift_exclude;
                    cv::Mat mat = (cv::Mat_<double>(2, 3) << 1, 0, neighbor_y_offset, 0, 1, neighbor_x_offset);
                    cv::warpAffine(target_bitmap.first, shift_threshold, mat, target_bitmap.first.size(),
                        cv::INTER_NEAREST);
                    cv::warpAffine(target_bitmap.second, shift_exclude, mat, target_bitmap.second.size(),
                        cv::INTER_NEAREST);

                    cv::Mat diff(standard_bitmap.first.size(), CV_8U);
                    cv::bitwise_xor(standard_bitmap.first, shift_threshold, diff);
                    cv::bitwise_and(diff, standard_bitmap.second, diff);
                    cv::bitwise_and(diff, shift_exclude, diff);

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
            cout << y_offset << ' ' << x_offset << endl;
            if (j != 0) {
                x_offset *= 2;
                y_offset *= 2;
            }
        }
        cout << "=======================final: " << y_offset << ' ' << x_offset <<endl;
        cv::Mat mat = (cv::Mat_<double>(2, 3) << 1, 0, y_offset, 0, 1, x_offset);
        cv::warpAffine(images[i], result[i], mat, images[i].size(),
            cv::INTER_NEAREST);
    }
    delete[] standard_pyramid;
    delete[] target_pyramid;
    return result;
}


int main() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    std::string folder = "./jingtong1/";
    int image_count = 10;
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
        //std::cout << time << '\n';
        exposure_times[i - 1] = te_interp(time.c_str(),0);
        std::cout << exposure_times[i - 1] << '\n';

    }
    //return 0;

    //int image_count = sizeof(exposure_times) / sizeof(exposure_times[0]); // P   
    //Mat* images = new Mat[image_count];
    std::vector<Mat> images;
    std::vector<tuple<Mat,double>> images_t;
   
    for (int i = 0; i < image_count; i++) {
        images.push_back(imread(image_paths[i], 1));
        images_t.push_back(make_tuple(imread(image_paths[i], IMREAD_COLOR), exposure_times[i]));
    }
    bool MTB_open = true;
    if (MTB_open) {
        //MTBA(images, images);
        images = MTB(images);
        //images_t = alignment(images_t,false);
        //for (int i = 0; i < images_t.size(); i++) {
        //    //cout << get<0>(images_t[i]).type() << endl;
        //    cvtColor(get<0>(images_t[i]),images[i],COLOR_BGRA2BGR);
        //    //images[i] = get<0>(images_t[i]);
        //}

    }
    for (int i = 0; i < images.size(); i++) 
    {
        //imshow("after" + to_string(i), images[i]);
        //imwrite("after" + to_string(i) +".jpg", images[i]);
    }
    //waitKey();
    //return 0;
    Mat* images_sample = new Mat[image_count];
   
    for (int i = 0; i < image_count; i++) {
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
    //imshow("HDR radiance", HDR_radiance);
    
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
    //Mat ldr;
    //Ptr<Tonemap> tonemap = createTonemap(2.2f);
    //tonemap->process(HDR, ldr);
    //imshow("opencv tonemapping", ldr);


    cout << "final img rows cols\n";

    cout << tonemapping.rows << ' ' << tonemapping.cols << endl;
    imshow("After tonemapping", tonemapping);

    //Mat tonemapping2;
    //Mat HDR_d(HDR.size(), CV_64FC3);
    //HDR.convertTo(HDR_d,CV_64FC3);
    //Mat output;
    //output = tone_mapping(HDR_d,0);
    //imshow("After tonemapping2", output);
    
    imwrite("tonemapping.png", tonemapping_8U);
    std::cout << "finish";
    cv::waitKey();
    return 0;
}