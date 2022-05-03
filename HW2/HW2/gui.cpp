#include "gui.h"
using namespace std;
using namespace cv;
void limit_img_size(Mat& img, int limit_size) {
	if (img.rows < limit_size && img.cols < limit_size) return;
	int new_rows, new_cols;
	if (img.cols > img.rows) {
		new_rows = limit_size * img.rows / img.cols;
		new_cols = limit_size;
	}
	else if (img.cols < img.rows) {
		new_cols = limit_size * img.cols / img.rows;
		new_rows = limit_size;
	}
	else {
		new_rows = limit_size;
		new_cols = limit_size;
	}
	resize(img, img, Size(new_cols, new_rows));
}


void run_gui() {
	vector<string> filenames;
	Mat result;
	GLuint result_img_id;

	glfwSetErrorCallback([](int error, const char* description) {fprintf(stderr, "Glfw Error %d: %s\n", error, description); });
	if (!glfwInit())
		return;

	int default_width = 1280, default_height = 800;
	GLFWwindow* window = glfwCreateWindow(default_width, default_height, "Panorama", NULL, NULL);
	int main_window_width = default_width, main_window_height = default_height;
	if (window == NULL)
		return;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	//glfwSetWindowSizeLimits(window, 1280, 720, GLFW_DONT_CARE, GLFW_DONT_CARE);
	bool err = gladLoadGL() == 0;
	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return;
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
	vector<Mat> img_show;
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
					ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose Images to Open", ".*,.png,.jpg,.PNG,.JPG", ".", 0);
				}
				if (ImGui::MenuItem("Save result to file"))
				{
					if (!result.empty()) {
						ImGuiFileDialog::Instance()->OpenDialog("ChooseFileSave", "Save File as", ".png,.jpg,.PNG,.JPG", ".", 0);
					}
				}
				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				filenames.clear();
				img_ids.clear();
				img_show.clear();
				map<string, string> selections = ImGuiFileDialog::Instance()->GetSelection();
				for (const auto& selection : selections) {
					filenames.push_back(selection.second);
					Mat image = imread(selection.second);
					limit_img_size(image, 300);
					cout << image.cols << " " << image.rows << endl;
					img_ids.push_back(TextureFromMat(image));
					img_show.push_back(image);
				}
			}
			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseFileSave"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				imwrite(ImGuiFileDialog::Instance()->GetFilePathName(), result);
			}
			ImGuiFileDialog::Instance()->Close();
		}


		{
			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 1.0f));
			ImVec2 scrolling_child_size = ImVec2(0, 320);
			ImGui::BeginChild("input_scrolling", scrolling_child_size, true, ImGuiWindowFlags_HorizontalScrollbar);
			for (int i = 0; i < img_ids.size(); i++) {
				ImGui::Image(ImTextureID(img_ids[i]), ImVec2(img_show[i].cols, img_show[i].rows));
				ImGui::SameLine();
			}
			float scroll_x = ImGui::GetScrollX();
			float scroll_max_x = ImGui::GetScrollMaxX();
			ImGui::EndChild();
		}

		if (filenames.size() > 0) {
			if (ImGui::Button("start image stitch", ImVec2(main_window_width, 50))) {
				result = image_stitch(filenames);
				cout << "image row col: " << result.rows << " " << result.cols << endl;
				imshow("panorama result", result);
				imwrite("result.png", result);
				cout << "complete" << endl;
				result_img_id = TextureFromMat(result);
			}
		}


		if (!result.empty()) {
			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 1.0f));
			ImVec2 scrolling_child_size = ImVec2(0, 0);
			ImGui::BeginChild("output_scrolling", scrolling_child_size, true, ImGuiWindowFlags_HorizontalScrollbar);
			ImGui::Image(ImTextureID(result_img_id), ImVec2(result.cols, result.rows));
			float scroll_x = ImGui::GetScrollX();
			float scroll_max_x = ImGui::GetScrollMaxX();
			ImGui::EndChild();
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
}