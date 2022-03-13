#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include <string>

#define WINDOW_NAME "HDR"
using namespace cv;
int main() {
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
    int image_count = sizeof(exposure_times) / sizeof(exposure_times[0]);
    Mat* images = new Mat[image_count];
    for (int i = 0; i < image_count; i++) {
        images[i] = imread(image_paths[i]);
        //cv::imshow(std::to_string(i), images[i]);
    }
    while (cv::waitKey(20) != 27);
    return 0;
}