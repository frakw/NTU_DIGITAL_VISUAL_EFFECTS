//#define EIGEN_USE_MKL_ALL // Determine if use MKL
//#define EIGEN_VECTORIZE_SSE4_2
#define _USE_MATH_DEFINES
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>
#define WINDOW_NAME "HDR"
int SAMPLE_N = 70;

#define Zfunc(img,COLOR,row,col) ((img).at<Vec3b>((row),(col))[(COLOR)])
//#define Wfunc(val) (val < 128.0f ? val+1.0f:256.0f-val)
#define Wfunc(val) (1.0f - ((float)abs(val-127)/127.0f))
using namespace cv;

// Traverse check
bool inside(int r, int c, int rows, int cols)
{
    return r >= 0 && r < rows&& c >= 0 && c < cols;
}
int main() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::string image_paths[] = {
        "./exposures/img01.jpg","./exposures/img02.jpg","./exposures/img03.jpg","./exposures/img04.jpg","./exposures/img05.jpg","./exposures/img06.jpg",
        "./exposures/img07.jpg","./exposures/img08.jpg","./exposures/img09.jpg","./exposures/img10.jpg","./exposures/img11.jpg","./exposures/img12.jpg",
        "./exposures/img13.jpg"
    };
    float exposure_times[] = {
        13.0f,10.0f,4.0f,3.2f,1.0f,0.8f,
        1.0f/3.0f,0.25f,1.0f/60.0f,1.0f/80.0f,1.0f/320.0f,1.0f/400.0f,
        1.0f/1000.0f
    };
    int image_count = sizeof(exposure_times) / sizeof(exposure_times[0]); // P   
    Mat* images = new Mat[image_count];


    Mat* images_sample = new Mat[image_count];
    int scale_down = 50;
    for (int i = 0; i < image_count; i++) {
        images[i] = imread(image_paths[i],1);
        resize(images[i], images_sample[i], Size(images[i].cols / scale_down, images[i].rows / scale_down));
        //cv::imshow(std::to_string(i), images[i]);
    }


    int pixel_count = images_sample[0].rows * images_sample[0].cols;// N
    int row_count = images_sample[0].rows;
    int col_count = images_sample[0].cols;

    int n = 256;
    float l = 40.0f;
    std::cout << pixel_count * image_count + n + 1 << std::endl;
    std::cout << row_count << ' ' << col_count << std::endl;

    Eigen::SparseMatrix<float> A(pixel_count * image_count + n + 1, n + pixel_count);
    //Eigen::MatrixXf A(pixel_count * image_count + n + 1, n + pixel_count);
    Eigen::VectorXf b(A.rows());



    Eigen::VectorXf x[3];
    float Gfunc[3][256];
    Mat HDR = Mat::zeros(images[0].size(), CV_32FC3);
    //Eigen::VectorXf x(n+ pixel_count);
    auto axes = CvPlot::makePlotAxes();
    #pragma omp parallel for
    for (int color = 0; color < 3; color++) {
        A.setZero();
        b.setZero();

        int k = 0;
        int i = 0;

        for (int row = 0; row < row_count; row++) {
            for (int col = 0; col < col_count; col++) {
                for (int img_index = 0; img_index < image_count; img_index++) {
                    int z_val = Zfunc(images_sample[img_index], color, row, col);
                    float wij = Wfunc(z_val);
                    A.coeffRef(k, z_val) = wij;
                    A.coeffRef(k, n + i) = -wij;
                    b.coeffRef(k) = wij * log(exposure_times[img_index]);
                    k = k + 1;
                }
                i++;
            }
        }
        A.coeffRef(k, 128) = 1.0f;
        k = k + 1;
        for (int i = 0; i < n - 2; i++) {
            float lW = l * Wfunc(i+1);
            A.coeffRef(k, i) = lW;
            A.coeffRef(k, i + 1) = -2 * lW;
            A.coeffRef(k, i + 2) = lW;
            k = k + 1;
        }



        std::cout << "Ax=b solving..." << std::endl;
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> lscg;
        lscg.compute(A);
        x[color] = lscg.solve(b);

        //x[color] = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        //Mat x;
        //solve(A, B, x, DECOMP_SVD); // Pseudo Inverse

        std::vector<float> resultG,x_index;
        for (int i = 0; i < 256; i++) {
            //Gfunc[color][i] = x[color].coeff(i);
            Gfunc[color][i] = exp(x[color](i));
            x_index.push_back(i);
        }
        
        switch (color)
        {
        case 0:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-r"); break;
        case 1:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-g"); break;
        case 2:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-b"); break;
        default:
            break;
        }


    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    CvPlot::show("response curve", axes);

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

                HDR.at<Vec3f>(row, col)(color) = exp(log_sum / weight_sum);
            }
        }
    }
    imshow("HDR", HDR);
    imwrite("HDR_Deb_image.exr", HDR);

    cv::waitKey();
    return 0;
}