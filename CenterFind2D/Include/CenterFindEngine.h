#ifndef CENTERFIND_ENGINE_H
#define CENTERFIND_ENGINE_H

#include <opencv2/opencv.hpp>
#include <FreeImage.h>

#include <set>
#include <array>
#include <vector>
#include <deque>

#include <stdint.h>

class CenterFindEngine
{
public:
	enum class OutputMode {
		TEXT,
		IMAGES,
		EVERY_100,
		LOG_COUNT
	};

	union ParticleMetrics {
		struct {
			float idx;
			float x_val;
			float y_val;
			float r2_val;
			float x_offset;
			float y_offset;
			float mass;
			float multiplicity;
		};
		float data[8];
		float& operator[](uint32_t idx) {
			return data[idx];
		}
	};

	//struct ParticleMetrics {
	//	union{
	//		float data[8];
	//		float x_offset;
	//		float y_offset
	//};

	//using ParticleMetrics = std::array<float, 8>;
	using PMetricsVec = std::vector<ParticleMetrics>;

	class Parameters {
	public:
		Parameters();
		// Only constructor, which takes in argv
		Parameters(std::array<std::string, 13> args);

		// General file name (without number)
		// i.e CenterfindData_in, CenterFindData_out, .tiff
		std::string m_strImgDir;
        std::string m_strInputStem;
		std::string m_strOutputStem;
		std::string m_strFileExt;
		uint32_t m_uFileNamePad;

		// First/Last stack number
		uint32_t m_uStartOfStack;
		uint32_t m_uEndOfStack;

		// First/Last frame within stack
		uint32_t m_uStartFrame;
		uint32_t m_uEndFrame;

		// Feature, Dilation, and Mask Radius
		// used during analysis
		uint32_t m_uFeatureRadius;
		uint32_t m_uDilationRadius;
		uint32_t m_uMaskRadius;

		// Halfwidth-Halfmodulation length
		// particle brightness threshold
		float m_fHWHMLength;
		float m_fPctleThreshold;

		// Various printing modes
		std::set<OutputMode> m_setOutputMode;

		bool IsOutputModeOn(OutputMode om);

		std::string GetFileName(uint32_t idx);
	};

	// The mat declarations represent a basic signal flow
	struct Data {
		cv::UMat m_InputImg;      // The input image
		cv::UMat m_BypassedImg;   // Bypass result
		cv::UMat m_ThresholdImg;  // Thresholded Bypass result
		cv::UMat m_LocalMaxImg;   // The local maximum image
		cv::UMat m_ParticleImg;   // The boolean image of particle locations
		PMetricsVec m_Data;       // The vector of particle data
	public:
		Data(FIBITMAP * bmp);
	};

	// The operators
	class ImgOperator {
		virtual void Execute(Data& data) = 0;
		virtual void operator()(Data& data) {
			Execute(data);
		}
	};
	// Bandpass
	class BandPass : public ImgOperator {
	public:
		BandPass();
		BandPass(int radius, float hwhm);
		void Execute(Data& data) override;
	private:
		uint32_t m_uGaussianRadius;
		cv::UMat m_GaussKernel;
		cv::UMat m_CircleMask;

	};

	// Local maximum
	class LocalMax : public ImgOperator {
	public:
		LocalMax();
		LocalMax(int radius, float pctl_thresh);
		void Execute(Data& data) override;
	private:
		uint32_t m_uDilationRadius;
		float m_fPctleThreshold;
		cv::UMat m_DilationKernel;
	};

	// Statistics
	class Statistics {
	public:
		Statistics();
		Statistics(int mask_radius, int feature_radius);
		PMetricsVec GetMetrics(Data& data);
	private:
		uint32_t m_uMaskRadius;
		uint32_t m_uFeatureRadius;
		cv::UMat m_CircleMask;
		cv::UMat m_RadXKernel; 
		cv::UMat m_RadYKernel;
		cv::UMat m_RadSqKernel;
		PMetricsVec m_Metrics;
	};

	// Now to the actual CenterFindEngine class
private:
	Parameters m_Params;
	std::vector<Data> m_Images;
	BandPass m_fnBandPass;
	LocalMax m_fnLocalMax;
	Statistics m_fnStatistics;

public:
	CenterFindEngine(const CenterFindEngine::Parameters params);
	std::deque<PMetricsVec> Execute();
};
//
//#include <vector>
//#include <deque>
//#include <sstream>
//#include <assert.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <FreeImage.h>
//
//#include <atomic>
//#include <condition_variable>
//#include <mutex>
//
//#define OCL_OCV
////#define CU_OCV
//
//using namespace std;
//using cv::MORPH_DILATE;
//using cv::THRESH_BINARY;
//using cv::THRESH_TOZERO;
//using cv::THRESH_TOZERO_INV;
//using cv::Mat;
//using cv::WINDOW_AUTOSIZE;
//using cv::waitKey;
//using cv::Rect;
//using cv::getGaussianKernel;
//using cv::getStructuringElement;
//using cv::MORPH_ELLIPSE;
//using cv::circle;
//
//using cv::imshow;
//
//#ifdef OCL_OCV
//#include <opencv2/ocl/ocl.hpp>
//
//using cv::ocl::createGaussianFilter_GPU;
//using cv::ocl::createMorphologyFilter_GPU;
//using cv::ocl::createLinearFilter_GPU;
//using cv::ocl::threshold;
//using cv::ocl::sepFilter2D;
//using cv::ocl::filter2D;
//using cv::ocl::dilate;
//using Image = cv::ocl::oclMat;
//using Filter = cv::Ptr < cv::ocl::FilterEngine_GPU > ;
////using Kernel = cv::ocl::oclMat;
//using Matrix = cv::ocl::oclMat;
//using cv::ocl::minMaxLoc;
//using cv::ocl::subtract;
//using cv::ocl::add;
//using cv::ocl::divide;
//using cv::ocl::multiply;
//using cv::ocl::exp;
//#elif defined CU_OCV
//#include <opencv2/gpu/gpu.hpp>
//
//using cv::gpu::createGaussianFilter_GPU;
//using cv::gpu::createMorphologyFilter_GPU;
//using cv::gpu::createLinearFilter_GPU;
//using cv::gpu::threshold;
//using cv::gpu::sepFilter2D;
//using cv::gpu::filter2D;
//using cv::gpu::dilate;
//using Image = cv::gpu::GpuMat;
//using Filter = cv::Ptr < cv::gpu::FilterEngine_GPU > ;
////using Kernel = cv::gpu::GpuMat;
//using Matrix = cv::gpu::GpuMat;
//using cv::gpu::minMaxLoc;
//using cv::gpu::subtract;
//using cv::gpu::add;
//using cv::gpu::divide;
//using cv::gpu::multiply;
//using cv::gpu::exp;
//#else
//using cv::createGaussianFilter;
//using cv::createMorphologyFilter;
//using cv::createLinearFilter;
//using cv::threshold;
//using cv::sepFilter2D;
//using cv::filter2D;
//using cv::dilate;
//using Image = cv::Mat;// _ < float > ;
//using Filter = cv::Ptr < cv::FilterEngine > ;
//using Matrix = cv::Mat;
//using cv::minMaxLoc;
//using cv::subtract;
//using cv::add;
//using cv::divide;
//using cv::multiply;
//using cv::exp;
//#endif
//
//class CenterFindEngine {
//
//	struct ParticleData{
//		float data[8];
//	};
//
//	struct Parameters{
//		using sstrm = stringstream;
//
//		string infile_stem;			//base for input file name, without extension
//		string outfile_stem;		//base for output file name, without extension
//		string file_extension;
//		int start_frameofstack;	//starting frame number in each stack
//		int end_frameofstack;	//ending frame number in each stack
//		int start_stack;			//starting stack number
//		int end_stack;			//total number of stacks to analyze
//		int feature_radius;		//characteristic radius of particles, in pixels
//		float hwhm_length;		//hwhm (half-width, half-maximum for gaussian kernel), in pixels
//		int dilation_radius;	//radius for dilation mask, in pixels
//		float pctle_threshold;	//percentile brightness to cutoff images
//		int mask_radius;		//radius for kernels for calculating mass, x and y positions, r^2, etc.
//		int testmode;			//execution modes: 0--print only text data; 1--output all intermediate images;
//		Parameters(){}
//		Parameters(string params[12]);
//		string getFileName(int idx, int pad = 4){
//			string num = to_string(idx);
//			while (num.length() < pad)
//				num = string("0") + num;
//			return infile_stem + "_" + num + file_extension;
//		}
//	} m_Params;
//
//	struct CenterFindData{
//		cv::Mat input;
//		cv::Mat_<unsigned char> particles;
//		Image in, bpass, bpass_thresh, local_max, tmp;
//		vector<ParticleData> m_Data;
//		condition_variable cv;
//		mutex m;
//		/*atomic_*/bool goodToGo{ false };
//		CenterFindData(const cv::Mat& m){
//			input = m;
//#ifndef CU_OCV
//			in = Image(m);
//#else
//			in.upload(m);
//#endif
//			tmp = Image({ in.rows, in.cols }, CV_32F, 0.f);
//			bpass = Image({ in.rows, in.cols }, CV_32F, 0.f);
//			bpass_thresh = Image({ in.rows, in.cols }, CV_32F, 0.f);
//			local_max = Image({ in.rows, in.cols }, CV_32F, 0.f);
//		}
//		CenterFindData(const CenterFindData& other) :
//			input(other.input),
//			in(other.in),
//			tmp(other.tmp),
//			bpass(other.bpass),
//			bpass_thresh(other.bpass_thresh),
//			local_max(other.local_max)
//		{}
//		int rows(){ return in.rows; }
//		int cols(){ return in.cols; }
//	};
//
//	struct BandPassEngine {
//		int m_Radius;
//		Filter Gaussian, Circle;
//		BandPassEngine(int radius = 1, float h = 1);
//		void operator()(CenterFindData& img);
//	} m_BandPass;
//
//	struct LocalMaxEngine {
//		int m_Radius;
//		float m_Pctl_Threshold;
//		Filter Dilation;
//		Image Threshold;
//		LocalMaxEngine(int radius = 1, float pctl_thresh = 1.f);
//		void operator()(CenterFindData& img);
//	} m_LocalMax;
//
//	struct StatisticsEngine {
//		int m_Mask_Radius, m_Feature_Radius;
//		cv::Mat_<float> Circle, rX, rY, r2;
//		StatisticsEngine(int mask_radius = 1, int feature_radius = 1);
//		vector<ParticleData> operator()(CenterFindData& img);
//	} m_Statistics;
//
//	vector<CenterFindData> m_Images;
//	vector<vector<ParticleData> > m_Data;
//	static void RecenterImage(Image& img, double range = 100.f);
//public:
//	CenterFindEngine(string params[12]);
//	static void showImage(Mat& img);
//};

#endif
