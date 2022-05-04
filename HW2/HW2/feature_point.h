#ifndef _FEATURE_POINT_H_
#define _FEATURE_POINT_H_
#include <vector>
#define FEATURE_ELEMENT_LENGTH 128
class FeaturePoint {
public:
	int    octave;
	int    interval;
	double offset_interval;
	int    x;
	int    y;
	double scale;
	double dx;
	double dy;
	double offset_x;
	double offset_y;
	double octave_scale;
	double ori;
	int    descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH]; 
	double val;
	std::vector<int> best_match;
	FeaturePoint() {}
	FeaturePoint(int _x,int _y,int _offset_x,int _offset_y,int _interval,int _offset_interval,int _octave)
		: x(_x),y(_y),offset_x(_offset_x),offset_y(_offset_y),interval(_interval),offset_interval(_offset_interval),octave(_octave)
	{
		dx = (x + offset_x) * pow(2.0, octave);
		dy = (y + offset_y) * pow(2.0, octave);
	}	
	FeaturePoint& operator=(const FeaturePoint& src) {
		this->dx = src.dx;
		this->dy = src.dy;

		this->interval = src.interval;
		this->octave = src.octave;
		this->octave_scale = src.octave_scale;
		this->offset_interval = src.offset_interval;

		this->offset_x = src.offset_x;
		this->offset_y = src.offset_y;

		this->ori = src.ori;
		this->scale = src.scale;
		this->val = src.val;
		this->x = src.x;
		this->y = src.y;
		return *this;
	}
};

#endif