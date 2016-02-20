#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>

#include <stdint.h>
#include <vector>
#include <list>
#include <memory>

using cv::cuda::GpuMat;

void RemapImage(GpuMat& img, float m, float M);
void DisplayImage(GpuMat& img);

// Forward for freeimage type
struct FIBITMAP;

struct Datum {
	int sliceIdx;
	GpuMat d_InputImg;      // The input image
	GpuMat d_FilteredImg;   // Filtered result
	GpuMat d_DilateImg;		// Dilated filtered result
	GpuMat d_LocalMaxImg;   // The local maximum image
	GpuMat d_ParticleImg;   // The boolean image of particle locations
	GpuMat d_ThreshImg;		// The thresholded filtered image, used in particle detection
	GpuMat d_TmpImg;		// Temp buffer

	Datum();
	Datum(const Datum& D); // This creates new data
	Datum(FIBITMAP * bmp, int sliceIdx);

	// Assignment impl of copy constructor
	Datum& operator=(const Datum& D);
};

// Abstract operator, could be useful later
class ImgOperator {
public:
	virtual void Execute(Datum& data) = 0;
	virtual void operator()(Datum& data) {
		Execute(data);
	}
};

// Bandpass
class BandPass : public ImgOperator {
public:
	BandPass();
	BandPass(int radius, float hwhm);
	void Execute(Datum& data) override;

	int GetGaussianRadius() const;
	float GetHalfWidthHalfModulation() const;
private:
	int m_uGaussianRadius;
	float m_fHWHM;
	cv::Ptr<cv::cuda::Filter> m_GaussFilter;
	cv::Ptr<cv::cuda::Filter> m_CircleFilter;

public:
	// set methods
	void SetGaussianRadius(float rad);
	void SetHWHM(float hwhm);
};

// Local maximum
class LocalMax : public ImgOperator {
public:
	LocalMax();
	LocalMax(int radius, float pctl_thresh);
	void Execute(Datum& data) override;

	int GetDilationRadius() const;
	float GetParticleThreshold() const;
private:
	int m_uDilationRadius;
	float m_fPctleThreshold;
	cv::Ptr<cv::cuda::Filter> m_DilationKernel;
	cv::Ptr<cv::cuda::Filter> m_DerivKernel;
public:
	// set methods
	void SetDilationRadius(int rad);
	void SetParticleThreshold(float pthresh);
};

// Because Solver has a thrust vector as a member it is forwarded here
class Solver;

class Engine {
	std::vector<Datum> m_vData;
	BandPass m_fnBandPass;
	LocalMax m_fnLocalMax;
	std::unique_ptr<Solver> m_ParticleSolver;

	void getUserParams( Datum D, BandPass * pEngineBP, LocalMax * pEngineLM );
public:
	Engine();
	~Engine();
	bool Init(std::list<std::string> liStackPaths, int startOfStack, int endOfStack);
	int Execute();

	BandPass * GetPandPass();
	LocalMax * GetLocalMax();
};

// Display functions
void showImage(cv::Mat& img);
void showImage(GpuMat& img);

// Function to upload cv::Mat to continuous GpuMat
GpuMat getContinuousGpuMat( cv::Mat& m );