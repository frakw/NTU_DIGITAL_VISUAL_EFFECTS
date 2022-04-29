#ifndef _SIFT_DEF_
#define _SIFT_DEF_


#define SIFT_SIGMA_MIN 0.8f

#define SIFT_MIN_PIX_DIST 0.5


//總共幾個(層)octave
#define SIFT_N_OCTAVE 8

// default number of sampled intervals per octave
//每層octave有幾個layer
#define SIFT_INTERVALS 3


#define SIFT_LAYER_PER_OCT  (SIFT_INTERVALS + 3)

#define SIFT_DOG_LAYER_PER_OCT  (SIFT_LAYER_PER_OCT - 1)

// default sigma for initial gaussian smoothing
#define SIFT_SIGMA 1.6f

#define SIFT_C_DOG 0.015f

#define SIFT_C_EDGE 10

#define SIFT_N_SPO 3

#define SIFT_MAX_REFINE_ITER 5

#define SIFT_LAMBDA_DESC 6.0f

#define SIFT_N_BINS  36

#define SIFT_LAMBDA_ORI  1.5f

#define SIFT_N_HIST 4

#define SIFT_N_ORI 8

// default threshold on keypoint contrast |D(x)|
#define SIFT_CONTR_THR 0.04f

// default threshold on keypoint ratio of principle curvatures
#define SIFT_CURV_THR 10.f

// double image size before pyramid construction?
#define SIFT_IMG_DBL true

// default width of descriptor histogram array
#define SIFT_DESCR_WIDTH 4

// default number of bins per histogram in descriptor array
#define SIFT_DESCR_HIST_BINS 8

// assumed gaussian blur for input image
#define SIFT_INIT_SIGMA 0.5f

// width of border in which to ignore keypoints
#define SIFT_IMG_BORDER 5

// maximum steps of keypoint interpolation before failure
#define SIFT_MAX_INTERP_STEPS 5

// default number of bins in histogram for orientation assignment
#define SIFT_ORI_HIST_BINS 36

// determines gaussian sigma for orientation assignment
#define SIFT_ORI_SIG_FCTR 1.5f

// determines the radius of the region used in orientation assignment
#define SIFT_ORI_RADIUS  (3 * SIFT_ORI_SIG_FCTR)

// orientation magnitude relative to max that results in new feature
#define SIFT_ORI_PEAK_RATIO 0.8f

// determines the size of a single descriptor orientation histogram
#define SIFT_DESCR_SCL_FCTR 3.f

// threshold on magnitude of elements of descriptor vector
#define SIFT_DESCR_MAG_THR 0.2f

// factor used to convert floating-point descriptor to unsigned char
#define SIFT_INT_DESCR_FCTR 512.f
#endif // !_SIFT_DEF_