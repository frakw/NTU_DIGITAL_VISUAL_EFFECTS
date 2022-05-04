#ifndef _GUI_H_
#define _GUI_H_
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "ImGuiFileDialog.h"

#include "image_stitch.h"

#include "texture.h"

void run_gui();

#endif