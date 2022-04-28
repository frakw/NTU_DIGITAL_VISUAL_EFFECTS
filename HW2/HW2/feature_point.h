#ifndef _FEATURE_POINT_H_
#define _FEATURE_POINT_H_
#include <vector>
using namespace std;
class FeaturePoint {
	
public:
	int row;
	int col;
	int octave; //哪個octave
	int layer_index;//在該octave中的位置(index)

	int x;
	int y;
	double sigma;
	double extremum_val;
	vector<uint8_t> descriptor;



	bool valid = false;
	FeaturePoint() {}
	FeaturePoint(int _row, int _col, int _octave, int _layer_index):row(_row),col(_col),octave(_octave),layer_index(_layer_index) {
		descriptor.reserve(128);
	}
};

#endif