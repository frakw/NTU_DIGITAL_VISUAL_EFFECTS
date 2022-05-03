#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "ImGuiFileDialog.h"

#include <ANN/ANN.h>

#include "image_stitch.h"
#include "sift.h"
#include "warping.h"
#include "texture.h"

using namespace cv;
using namespace std;


int main(int argc,char* argv[]) {
	vector<string> filenames;
	Mat result;
	GLuint result_img_id;
	if (argc > 1) {
		fstream img_list_file(argv[1]);
		string line;
		while (getline(img_list_file, line)) {
			if (line.empty()) continue;
			filenames.push_back(line);
		}
		result = image_stitch(filenames);
		imshow("panorama result", result);
		imwrite("result.png", result);
		return 0;
	}

	srand(time(NULL));
	glfwSetErrorCallback([](int error, const char* description) {fprintf(stderr, "Glfw Error %d: %s\n", error, description); });
	if (!glfwInit())
		return 1;

	GLFWwindow* window = glfwCreateWindow(1280, 720, "Panorama", NULL, NULL);
	int main_window_width = 1280, main_window_height = 720;
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
	//ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".*,.png,.jpg,.PNG,.JPG", ".", 0);
	vector<int> img_ids;
	//ImGui::SetWindowFontScale(1.8f);
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		glfwGetWindowSize(window, &main_window_width, &main_window_height);
		ImGui::SetNextWindowPos(ImVec2(0, 0), 0, ImVec2(0, 0));
		ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y));
		ImGui::SetNextWindowBgAlpha(0);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
		ImGui::Begin("background", NULL, ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoBringToFrontOnFocus |
			//ImGuiWindowFlags_NoInputs |
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoResize);
			//ImGuiWindowFlags_NoScrollbar);

		//ImGui::SetNextItemWidth
		//ImGui::SetNextItemWidth(1280);
		//ImGui::Begin("test");

		ImGui::SetNextWindowContentSize(ImVec2(main_window_width, main_window_height));

		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Load image files"))
				{
					ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".*,.png,.jpg,.PNG,.JPG", ".", 0);
				}
				if (ImGui::MenuItem("Save result to file"))
				{

				}
				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
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


		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 1.0f));
		ImVec2 scrolling_child_size = ImVec2(0, ImGui::GetFrameHeightWithSpacing() * 7 + 30);
		ImGui::BeginChild("scrolling", scrolling_child_size, true, ImGuiWindowFlags_HorizontalScrollbar);
		for (int i = 0; i < img_ids.size(); i++) {
			ImGui::Image(ImTextureID(img_ids[i]), ImVec2(200, 200));
			ImGui::SameLine();
		}
		float scroll_x = ImGui::GetScrollX();
		float scroll_max_x = ImGui::GetScrollMaxX();
		ImGui::EndChild();

		ImGui::NewLine();
		if (filenames.size() > 0) {
			if (ImGui::Button("start image stitch",ImVec2(main_window_width,50))) {
				result = image_stitch(filenames);
				cout << "image row col: " << result.rows << " " << result.cols << endl;
				imshow("panorama result", result);
				imwrite("result.png", result);
				cout << "complete" << endl;
				result_img_id = TextureFromMat(result);
				//waitKey(0);
			}
		}

		if (!result.empty()) {
			ImGui::Image(ImTextureID(result_img_id), ImVec2(result.cols, result.rows));
		}
		



		ImGui::End();



		ImGui::ShowDemoWindow();

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


	
	return 0;
}