#include "image_stitch.h"

Mat image_stitch(vector<string> filenames) {
	Mat result;
	vector<Mat> imgs(filenames.size());
	for (int i = 0; i < filenames.size(); i++) {
		imgs[i] = imread(filenames[i]);
	}
	vector<vector<FeaturePoint>> img_fps_list(filenames.size());
	for (int i = 0; i < filenames.size(); i++) {
		img_fps_list[i] = SIFT(imgs[i]);
	}
	featureMatch(img_fps_list);
	//for (int i = 0; i < img_fps_list.size(); i++) {
	//	for (int j = 0; j < img_fps_list[i].size(); j++) {
	//		cout << "size:" << img_fps_list[i][j].best_match.size() << " [0]: "<< img_fps_list[i][j].best_match[0] << " [1]: " << img_fps_list[i][j].best_match[1] <<endl;
	//	}
	//	cout << "next img:" << endl;
	//}


	//Mat match_result = draw_matches2(imgs[0], imgs[1], img_fps_list[0], img_fps_list[1]);
	//imshow("match", match_result);
	imshow("feature point", draw_keypoints(imgs[1], img_fps_list[1],3));
	return result;
}