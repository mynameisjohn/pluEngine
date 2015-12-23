#pragma once

#include <FreeImage.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>

#include <stdint.h>
#include <vector>
#include <list>

using cv::cuda::GpuMat;

void RemapImage(GpuMat& img, float m, float M);
void DisplayImage(GpuMat& img);

struct Particle {
	float x;
	float y;
	float z;
	float i;
};

class ParticleStack {
	uint32_t m_uParticleCount;
	uint32_t m_uLastSliceIdx;
	float m_fMaxPeak;
	std::list<Particle> m_liContributingParticles;
public:
	ParticleStack();
	ParticleStack(Particle first, uint32_t sliceIdx);
	uint32_t AddParticle(Particle p, uint32_t sliceIdx);
	Particle GetRefinedParticle() const;
	uint32_t GetParticleCount() const;
	Particle GetLastParticleAdded() const;
	float GetPeak() const;
	uint32_t GetLastSliceIdx() const;

	// Comparison operator
	struct comp {
		bool operator()(const ParticleStack& a, const ParticleStack& b);
	};
};

struct Datum {
	int sliceIdx;
	GpuMat d_InputImg;      // The input image
	GpuMat d_FilteredImg;   // Filtered result
	GpuMat d_DilateImg;		// Dilated filtered result
	GpuMat d_LocalMaxImg;   // The local maximum image
	GpuMat d_ParticleImg;   // The boolean image of particle locations
	GpuMat d_TmpImg;		// The boolean image of particle locations

	Datum();
	Datum(const Datum& D); // This creates new data
	Datum(FIBITMAP * bmp, uint32_t sliceIdx);

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
private:
	uint32_t m_uGaussianRadius;
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
private:
	uint32_t m_uDilationRadius;
	float m_fPctleThreshold;
	cv::Ptr<cv::cuda::Filter> m_DilationKernel;

public:
	// set methods
	void SetDilationRadius(uint32_t rad);
	void SetParticleThreshold(float pthresh);
};

class Solver {
public:
	Solver();
	Solver(uint32_t mR, uint32_t fR, uint32_t minSC, uint32_t maxSC, uint32_t nR);
	uint32_t FindParticles(Datum& D);
	std::vector<Particle> GetFoundParticles() const;
private:
	uint32_t m_uMaskRadius;
	uint32_t m_uFeatureRadius;
	uint32_t m_uMinStackCount;
	uint32_t m_uMaxStackCount;
	uint32_t m_uNeighborRadius;
	cv::Mat m_CircleMask; // on the host for now
	cv::Mat m_RadXKernel;
	cv::Mat m_RadYKernel;
	cv::Mat m_RadSqKernel;
	std::vector<ParticleStack> m_vFoundParticles;
};

class Engine {
	std::vector<Datum> m_vData;
	BandPass m_fnBandPass;
	LocalMax m_fnLocalMax;
	Solver m_ParticleSolver;
public:
	Engine();
	bool Init(std::string stackPath, int startOfStack, int endOfStack);
	int Execute();

	BandPass * GetPandPass();
	LocalMax * GetLocalMax();
};