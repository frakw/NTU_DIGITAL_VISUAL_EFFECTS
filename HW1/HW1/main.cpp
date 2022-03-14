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
#include <string>
#include <cmath>
#include <chrono>
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>
#define WINDOW_NAME "HDR"

#define Zfunc(img,COLOR,row,col) ((img).at<Vec3b>((row),(col))[(COLOR)])
#define Wfunc(val) (1.0f - ((float)abs(val-127)/127.0f))
using namespace cv;
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
    int scale_down = 20;
    for (int i = 0; i < image_count; i++) {
        images[i] = imread(image_paths[i],CV_8UC3);
        resize(images[i], images[i], Size(images[i].cols / scale_down, images[i].rows / scale_down));
        //resize(images[i], images[i], Size(10,10));
        //cv::imshow(std::to_string(i), images[i]);
    }
    int pixel_count = images[0].rows * images[0].cols;// N
    int row_count = images[0].rows;
    int col_count = images[0].cols;
    int n = 256;
    float l = 40.0f;
    std::cout << pixel_count * image_count + n + 1 << std::endl;
    std::cout << row_count << ' ' << col_count << std::endl;
    //Eigen::SparseMatrix<float> A(pixel_count * image_count + n + 1, n + pixel_count);
    Eigen::MatrixXf A(pixel_count * image_count + n + 1, n + pixel_count);
    
    Eigen::VectorXf b(A.rows());
    //Eigen::VectorXf x(n+ pixel_count);
    auto axes = CvPlot::makePlotAxes();
    for (int color = 0; color < 3; color++) {
        A.setZero();
        b.setZero();
        int k = 0;
        int i = 0;
        for (int row = 0; row < images[0].rows; row++) {
            for (int col = 0; col < images[0].cols; col++) {
                for (int img_index = 0; img_index < image_count; img_index++) {
                    int z_val = Zfunc(images[img_index], color, row, col);
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

        //Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float> > lscg;
        //lscg.compute(A);
        //Eigen::VectorXf x = lscg.solve(b);

        std::cout << "SVD is on progress..." << std::endl;
        Eigen::VectorXf x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        std::cout << x.rows() << std::endl;
        std::vector<float> resultG,x_index;
        for (int i = 0; i < 256; i++) {
            //if (abs(x.coeff(i)) < 10.0f) {
            resultG.push_back(x.coeff(i));
            //}
            x_index.push_back(i);
        }
        
        switch (color)
        {
        case 0:axes.create<CvPlot::Series>(x_index, resultG, "-r"); break;
        case 1:axes.create<CvPlot::Series>(x_index, resultG, "-g"); break;
        case 2:axes.create<CvPlot::Series>(x_index, resultG, "-b"); break;
        default:
            break;
        }


    }

    CvPlot::show("mywindow", axes);

    std::cout << "finish"<< std::endl;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    cv::waitKey();
    return 0;
}