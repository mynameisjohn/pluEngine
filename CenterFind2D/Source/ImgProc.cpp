#include "CenterFind.h"

#include "FnPtrHelper.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <FreeImage.h>

// I may make this adjustable at some point
const static float kEpsilon = 0.0001f;

void showImage(cv::Mat& img) {
	using namespace cv;
	Mat x = img;

	//RecenterImage(x, 1);
	double min(1), max(2);
	minMaxLoc(x, &min, &max);
	x = (x - min) / (max - min);

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", x);                   // Show our image inside it.
	waitKey(0);
}

void showImage(GpuMat& img) {
	cv::Mat m;
	img.download(m);
	showImage(m);
}

Datum::Datum() :
	sliceIdx(0) 
{}

Datum::Datum(FIBITMAP * bmp, uint32_t sliceIdx) :
sliceIdx(sliceIdx) {
	// Create 24-bit RGB image, initialize to zero
	cv::Mat image = cv::Mat::zeros(FreeImage_GetWidth(bmp), FreeImage_GetHeight(bmp), CV_8UC3);

	// Convert FIBITMAP to 24 bit RGB, store inside cv::Mat's buffer
	FreeImage_ConvertToRawBits(image.data, bmp, image.step, 24, 0xFF, 0xFF, 0xFF, true);

	// Convert cvMat to greyscale float
	cvtColor(image, image, CV_RGB2GRAY);
	
	image.convertTo(image, CV_32F);
	image /= 255.f;
	
	// Upload input to device
	d_InputImg.upload(image);
	// Initialize all other mats to zero on device (may be superfluous)
	d_FilteredImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);
	d_DilateImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);	
	d_TmpImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);

	// It's in our best interest to ensure these are continuous
	cv::cuda::createContinuous( d_InputImg.size(), CV_32F, d_LocalMaxImg );
	d_LocalMaxImg.setTo( 0.f );

	cv::cuda::createContinuous( d_InputImg.size(), CV_8U, d_ParticleImg);
	d_ParticleImg.setTo(0);
}

// Copy all new data
Datum::Datum(const Datum& D){
	// Can I just do this?
	*this = D;
}

Datum& Datum::operator=(const Datum& D) {
	this->sliceIdx = D.sliceIdx;
	D.d_InputImg.copyTo(this->d_InputImg);
	D.d_FilteredImg.copyTo(this->d_FilteredImg);
	D.d_DilateImg.copyTo(this->d_DilateImg);
	D.d_LocalMaxImg.copyTo(this->d_LocalMaxImg);
	D.d_ParticleImg.copyTo(this->d_ParticleImg);
	D.d_TmpImg.copyTo(this->d_TmpImg);

	return *this;
}

BandPass::BandPass() :
ImgOperator(),
m_uGaussianRadius(0),
m_fHWHM(0.f) {
}

BandPass::BandPass(int radius, float hwhm) :
ImgOperator(),
m_uGaussianRadius(radius),
m_fHWHM(hwhm) {
	// Create circle image
	uint32_t diameter = 2 * m_uGaussianRadius + 1;
	cv::Mat h_Circle = cv::Mat::zeros(cv::Size(diameter, diameter), CV_32F);
	cv::circle(h_Circle, cv::Size(m_uGaussianRadius, m_uGaussianRadius), m_uGaussianRadius, 1.f, -1);

	// Upload to device
	m_CircleFilter = cv::cuda::createLinearFilter(CV_32F, CV_32F, h_Circle);

	// Create Gaussian Filter
	const cv::Size filterDiameter(diameter, diameter);
	const double sigma = m_fHWHM / ((sqrt(2 * log(2))));
	m_GaussFilter = cv::cuda::createGaussianFilter(CV_32F, CV_32F, filterDiameter, sigma);
}

void BandPass::Execute(Datum& D) {
	// Make some references
	GpuMat& in = D.d_InputImg;
	GpuMat& bp = D.d_FilteredImg;
	GpuMat& tmp = D.d_TmpImg;

	// Apply Gaussian filter
	m_GaussFilter->apply(in, bp);

	// Apply linear circle filter, store in tmp
	m_CircleFilter->apply(in, tmp);

	// scale tmp down
	const double scale = 1. / (3 * pow(m_uGaussianRadius, 2));
	tmp.convertTo(tmp, CV_32F, scale);

	// subtract tmp from bandpass to get filtered output
	cv::cuda::subtract(bp, tmp, bp);

	// Any negative values become 0
	cv::cuda::threshold(bp, bp, 0, 1, cv::THRESH_TOZERO);
}

uint32_t BandPass::GetGaussianRadius() const
{
	return m_uGaussianRadius;
}

float BandPass::GetHalfWidthHalfModulation() const
{
	return m_fHWHM;
}

LocalMax::LocalMax() :
ImgOperator(),
m_uDilationRadius(0),
m_fPctleThreshold(0) {
}

LocalMax::LocalMax(int radius, float pctl_thresh) :
ImgOperator(),
m_uDilationRadius(radius),
m_fPctleThreshold(pctl_thresh) {
	// Create dilation mask
	uint32_t diameter = 2 * m_uDilationRadius + 1;
	cv::Mat h_Dilation = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(diameter, diameter));

	// Create dilation kernel from host kernel (only single byte supported? why nVidia why)
	m_DilationKernel = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_32F, h_Dilation);

	// might need to mess with normalization and scale
	m_DerivKernel = cv::cuda::createDerivFilter(CV_32F, CV_32F, 1, 1, diameter, true);
}

void LocalMax::Execute(Datum& D) {
	// Make some references
	GpuMat& bp = D.d_FilteredImg;
	GpuMat& dil = D.d_DilateImg;
	GpuMat& lm = D.d_LocalMaxImg;
	GpuMat& tmp = D.d_TmpImg;
	GpuMat& thresh = D.d_ThreshImg;
	GpuMat out;

	thresh = GpuMat( bp.size(), CV_32F, (uint8_t) m_fPctleThreshold );
	RemapImage( bp, 0, 100 );
	cv::cuda::max( bp, thresh, thresh );

	dil.setTo( m_fPctleThreshold );
	m_DilationKernel->apply( thresh, dil );
	cv::cuda::subtract( bp, dil, lm );
	cv::cuda::exp( lm, lm );
	cv::cuda::threshold( lm, lm, 1 - kEpsilon, 1, cv::THRESH_BINARY );
	lm.convertTo( D.d_ParticleImg, CV_8U );
}

uint32_t LocalMax::GetDilationRadius() const
{
	return m_uDilationRadius;
}

float LocalMax::GetParticleThreshold() const
{
	return m_fPctleThreshold;
}