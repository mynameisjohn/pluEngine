#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudev.hpp>
#include <opencv2/core/cuda.hpp>

#include <FreeImage.h>

#include <list>
#include <vector>

#include <stdint.h>

namespace CenterFind
{
	using cv::cuda::GpuMat;

	struct Particle {
		float x{ -1 };    // x position (px)
		float y{ -1 };    // y position (px)
		float z{ -1 };    // z position (?)
		float i{ -1 };    // intensity (?)
	};

	class PStack {
		// We use this to manage intensity state (should increase then decrease)
		enum class IState{
			NONE,
			INCREASING,
			DECREASING,
			SEVER
		};

	private:
		uint32_t m_uParticleCount;
		uint32_t m_uLastSliceIdx;
		float m_fMaxPeak;
		IState m_CurIntensityState;
		std::list<Particle> m_liParticles;
	public:
		PStack();
		PStack(Particle first, uint32_t slice);
		void AddParticle(Particle p, uint32_t slice);
		Particle GetRefinedParticle() const;
		uint32_t GetParticleCount() const;
		Particle GetLastParticleAdded() const;
		float GetPeak() const;
		uint32_t GetLastSliceIdx() const;
		IState GetCurIntensityState() const;
		void AdvanceIntensityState();
		void Collapse();

		// Iterator functions
		auto begin() -> decltype(m_liParticles.begin()) {
			return m_liParticles.begin();
		}
		auto end() -> decltype(m_liParticles.end()) {
			return m_liParticles.end();
		}
	};

	// The mat declarations represent a basic signal flow
	struct Datum {
		uint32_t uSliceIdx;
		GpuMat d_InputImg;      // The input image
		GpuMat d_FilteredImg;   // Filtered result
		GpuMat d_ThresholdImg;  // Thresholded filtered result
		GpuMat d_LocalMaxImg;   // The local maximum image
		GpuMat d_ParticleImg;   // The boolean image of particle locations
		GpuMat d_TmpImg;		// The boolean image of particle locations

		Datum();
		Datum(FIBITMAP * bmp, uint32_t sliceIdx);
	};

	// Abstract operator, could be useful later
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
		void Execute(Datum& data) override;
	private:
		uint32_t m_uGaussianRadius;
		float m_fHWHM;
		cv::Ptr<cv::cuda::Filter> m_GaussFilter;
		cv::Ptr<cv::cuda::Filter> m_CircleFilter;
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
	};

	class Solver {
	public:
		Solver();
		Solver(int mask_radius, int feature_radius, int max_stack_count, float neighbor_radius);
		uint32_t FindParticles(Data& data);
		std::vector<Particle> GetFoundParticles();
	private:
		uint32_t m_uMaskRadius;
		uint32_t m_uFeatureRadius;
		uint32_t m_uMaxStackCount;
		float m_fNeighborRadius;
		cv::UMat m_CircleMask;
		cv::UMat m_RadXKernel;
		cv::UMat m_RadYKernel;
		cv::UMat m_RadSqKernel;
		std::vector<PStack> m_vFoundParticles;
	};

	class Engine {
		std::vector<Data> m_vImages;
		BandPass m_fnBandPass;
		LocalMax m_fnLocalMax;
		Solver m_ParticleSolver;
	public:
		Engine();
	};
}