#ifndef _TEXTURE_H_
#define _TEXTURE_H_
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <glad/glad.h>

using namespace std;
using namespace cv;
struct Texture {
    unsigned int id;
    string type;
    string path;
};
unsigned int TextureFromMat(const Mat& img);
//unsigned int TextureFromFile(const char* path, bool gamma = false);
#endif // !_TEXTURE_H_