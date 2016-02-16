#include "CenterFind.h"
#include "FnPtrHelper.h"
#include "Solver.cuh"

#include <opencv2/cudaarithm.hpp>

#include <map>

#include <FreeImage.h>

Engine::Engine() {
}

// Must be implemented here to know what a Solver is (and destroy it)
Engine::~Engine()
{

}

int Engine::Execute() {
	// Return if no data
	if (m_vData.empty())
		return -1;

	// This lets the user set DSP params
	getUserParams(m_vData.front(), &m_fnBandPass, &m_fnLocalMax );

	m_ParticleSolver = std::unique_ptr<Solver>( new Solver( 3, m_fnBandPass.GetGaussianRadius(), 3, 5, 8 ) );

	// Run Centerfind algorithm on remaining images
	for (auto& D : m_vData) {
		m_fnBandPass(D);
		m_fnLocalMax(D);
		m_ParticleSolver->FindParticles(D);
	}

	//auto shit = m_ParticleSolver->GetFoundParticles();
	return 0;
}


void RemapImage(GpuMat& img, float m, float M) {
	float range = M - m;
	double min(1), max(2);
	cv::cuda::minMax(img, &min, &max);
	double alpha = range / (max - min);
	double beta = range * min / (max - min);
	double scale = range / (max - min);
	img.convertTo(img, CV_32F, alpha, beta);
}

void DisplayImage(GpuMat& img) {
	GpuMat disp;
	img.convertTo(disp, CV_32F);
	RemapImage(disp, 0, 1);
	std::string winName("disp");
	cv::namedWindow(winName, cv::WINDOW_OPENGL);
	cv::imshow(winName, disp);
	cv::waitKey();
}