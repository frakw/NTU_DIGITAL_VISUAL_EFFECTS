#define _CRT_SECURE_NO_WARNINGS
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "ImGuiFileDialog.h"

#include <opencv2/opencv.hpp>
#include <ANN/ANN.h>
#include "image_stitch.h"
#include "sift.h"
#include "warping.h"
#include "exif.h"
#include "texture.h"

using namespace cv;
using namespace std;


int main(int argc,char* argv[]) {
	vector<string> filenames;


	srand(time(NULL));
	glfwSetErrorCallback([](int error, const char* description) {fprintf(stderr, "Glfw Error %d: %s\n", error, description); });
	if (!glfwInit())
		return 1;

	GLFWwindow* window = glfwCreateWindow(1280, 720, "Panorama", NULL, NULL);
	if (window == NULL)
		return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	//glfwSetWindowSizeLimits(window, 1280, 720, GLFW_DONT_CARE, GLFW_DONT_CARE);
	bool err = gladLoadGL() == 0;
	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return 1;
	}
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
	ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".*,.png,.jpg,.PNG,.JPG", ".", 0);
	vector<int> img_ids;
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		//ImGui::SetNextItemWidth
		ImGui::SetNextItemWidth(1280);
		ImGui::Begin("test");
		if (ImGui::Button("load image files")) {
			ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".*,.png,.jpg,.PNG,.JPG", ".", 0);
		}
		// display
		if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
		{
			// action if OK
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				filenames.clear();
				img_ids.clear();
				map<string, string> selections = ImGuiFileDialog::Instance()->GetSelection();
				for (const auto& selection : selections) {
					filenames.push_back(selection.second);
					Mat image = imread(selection.second);
					img_ids.push_back(TextureFromMat(image));
				}
			}

			// close
			ImGuiFileDialog::Instance()->Close();
		}
		if (ImGui::Button("start image stitch") && !filenames.empty()) {
			Mat result = image_stitch(filenames);
			cout << "image row col: " << result.rows << " " << result.cols << endl;
			imshow("panorama result", result);
			imwrite("result.png", result);
			cout << "complete" << endl;
			//waitKey(0);
		}
		for (int i = 0; i < img_ids.size(); i++) {
			ImGui::Image(ImTextureID(img_ids[i]), ImVec2(200, 200));
			ImGui::SameLine();
		}
		ImGui::End();

		ImGui::Render();
		int display_w, display_h;
		glfwMakeContextCurrent(window);
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwMakeContextCurrent(window);
		glfwSwapBuffers(window);
		glUseProgram(0);
		glColor3b(100, 100, 100);
		glClear(GL_COLOR_BUFFER_BIT);
		glFlush();
	}


	//vector<string> filenames =
	//{ 
	//	"./test_image/parrington/prtn08.jpg" ,
	//	"./test_image/parrington/prtn01.jpg" ,
	//	"./test_image/parrington/prtn02.jpg" ,
	//	"./test_image/parrington/prtn03.jpg" ,
	//	"./test_image/parrington/prtn04.jpg" ,
	//	"./test_image/parrington/prtn05.jpg" ,
	//	"./test_image/parrington/prtn06.jpg" ,
	//	"./test_image/parrington/prtn17.jpg" ,
	//	"./test_image/parrington/prtn00.jpg" ,
	//	"./test_image/parrington/prtn09.jpg" ,
	//	"./test_image/parrington/prtn10.jpg" ,
	//	"./test_image/parrington/prtn11.jpg" ,
	//	"./test_image/parrington/prtn12.jpg" ,
	//	"./test_image/parrington/prtn13.jpg" ,
	//	"./test_image/parrington/prtn14.jpg" ,
	//	"./test_image/parrington/prtn15.jpg" ,
	//	"./test_image/parrington/prtn16.jpg" ,
	//	"./test_image/parrington/prtn07.jpg" ,
	//};
	// 
	 
	
	//vector<string> filenames =
	//{
	//	"./test_image/grail/grail04.jpg" ,
	//	"./test_image/grail/grail02.jpg" ,
	//	"./test_image/grail/grail01.jpg" ,
	//	"./test_image/grail/grail16.jpg" ,
	//	"./test_image/grail/grail15.jpg" ,
	//	"./test_image/grail/grail17.jpg" ,
	//	"./test_image/grail/grail00.jpg" ,
	//	"./test_image/grail/grail03.jpg" ,
	//};

	//vector<string> filenames =
	//{
	//	"./my_pic/1/P_20220430_173030.jpg" ,
	//	"./my_pic/1/P_20220430_173041.jpg" ,
	//	"./my_pic/1/P_20220430_173035.jpg" ,
	//	"./my_pic/1/P_20220430_173050.jpg" ,
	//	"./my_pic/1/P_20220430_173046.jpg" ,
	//	"./my_pic/1/P_20220430_173052.jpg" ,
	//	"./my_pic/1/P_20220430_173038.jpg" ,
	//	"./my_pic/1/P_20220430_173043.jpg" ,
	//	"./my_pic/1/P_20220430_173106.jpg" ,
	//	"./my_pic/1/P_20220430_173057.jpg" ,
	//};

	//	vector<string> filenames =
	//{
	//	"./my_pic/3/P_20211115_174223.jpg" ,
	//	"./my_pic/3/P_20211115_174231.jpg" ,
	//	"./my_pic/3/P_20211115_174238.jpg" ,
	//	"./my_pic/3/P_20211115_174244.jpg" ,
	//};

	//vector<string> filenames =
	//{
	//	"./my_pic/2/DSC00020.jpg" ,
	//	"./my_pic/2/DSC00021.jpg" ,
	//	"./my_pic/2/DSC00022.jpg" ,
	//	"./my_pic/2/DSC00023.jpg" ,
	//	"./my_pic/2/DSC00024.jpg" ,
	//	"./my_pic/2/DSC00025.jpg" ,
	//	"./my_pic/2/DSC00026.jpg" ,
	//	"./my_pic/2/DSC00027.jpg" ,
	//	"./my_pic/2/DSC00028.jpg" ,
	//	"./my_pic/2/DSC00029.jpg" ,
	//	"./my_pic/2/DSC00030.jpg" ,
	//	"./my_pic/2/DSC00031.jpg" ,
	//	"./my_pic/2/DSC00032.jpg" ,
	//	"./my_pic/2/DSC00033.jpg" ,
	//	"./my_pic/2/DSC00034.jpg" ,
	//	"./my_pic/2/DSC00035.jpg" ,
	//	"./my_pic/2/DSC00036.jpg" ,
	//	"./my_pic/2/DSC00037.jpg" ,
	//	"./my_pic/2/DSC00038.jpg" ,
	//	"./my_pic/2/DSC00039.jpg" ,
	//	"./my_pic/2/DSC00040.jpg" ,
	//};


	//Mat img1 = imread("./Lenna.jpg");
	//Mat img2 = imread("./Lenna_rotate_scale.png");
	//Mat img = imread("./Lenna_rotate_scale.png");
	//Mat img = imread("./Lenna_rotate_scale1.png");
	//GaussianBlur(img, img, Size(3, 3), 1.6f, 1.6f);
	//vector<FeaturePoint> fps1 = SIFT(img1);
	//vector<FeaturePoint> fps2 = SIFT(img2);

	//std::vector<std::pair<int, int>> matches = find_keypoint_matches(fps1, fps2);
	//Mat result = draw_matches(img1, img2, fps1, fps2, matches);
	//imshow("match", result);


	
	return 0;
}


/*

	//FILE* fp = fopen("./square00.jpg", "rb");
	//fseek(fp, 0, SEEK_END);
	//unsigned long fsize = ftell(fp);
	//rewind(fp);
	//unsigned char* buf = new unsigned char[fsize];
	//if (fread(buf, 1, fsize, fp) != fsize) {
	//	printf("Can't read file.\n");
	//	delete[] buf;
	//	return -2;
	//}
	//fclose(fp);

	//// Parse EXIF
	//easyexif::EXIFInfo result;
	//int code = result.parseFrom(buf, fsize);
	//delete[] buf;
	//if (code) {
	//	printf("Error parsing EXIF: code %d\n", code);
	//	return -3;
	//}
	//cout << "focal len: " << result.FocalLengthIn35mm << endl;
	Mat img1 = imread("./P_20220430_174302.jpg");
	resize(img1, img1, Size(img1.cols / 3, img1.rows / 3));
	double f = ((int)img1.rows / 10) * 10;
	vector<FeaturePoint> tmp;
	imshow("warping", cylindrical_warping2(img1, tmp, f));
	waitKey(0);
	return 0;
*/