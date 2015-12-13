#include "CenterFind.h"

using namespace Centerfind;

Datum::Datum() :
m_uSliceIdx(0) {
}

Datum::Datum(FIBITMAP * bmp, uint32_t sliceIdx) :
uSliceIdx(sliceIdx) {
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

	// Initialize all other mats to zero on device
	d_FilteredImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);
	d_ThresholdImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);
	d_LocalMaxImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);
	d_ParticleImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);
	d_TmpImg = GpuMat(d_InputImg.size(), CV_32F, 0.f);
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
	const double sigma = 0.5 * (m_fHWHM / 0.8325546);
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
	const float scale = 0.33333f * powf(m_uGaussianRadius, 2);
	tmp.convertTo(tmp, CV_32F, scale);

	// subtract tmp from bandpass to get filtered output
	cv::cuda::subtract(bp, tmp, bp);

	// Any negative values become 0
	cv::cuda::threshold(bp, bp, 0, 1, cv::THRESH_TOZERO);
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
	m_DilationKernel = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8U, h_Dilation);
}

void LocalMax::Execute(Datum& D) {
	// Useful constants
	const double kEPS(0.0000001f);				// A low number
	const double kToSingleByte = double(0xFF);	// brings float[0, 1] to byte[0, 255]
	const double kToFloat = 1 / double(0xFF);	// and back again

	// Make some references
	GpuMat& bp = D.d_FilteredImg;
	GpuMat& bpt = D.d_ThresholdImg;
	GpuMat& lm = D.d_LocalMaxImg;
	GpuMat& tmp = D.d_TmpImg;

	// Assign entire bp thresh image to particle threshold
	bpt.setTo(m_fPctleThreshold);

	// I have no idea why, but Peter remaps the image between 0 and 100 here
	double range(100);
	double min(1), max(2);
	cv::cuda::minMax(bp, &min, &max);
	double alpha = range / (max - min);
	double beta = range * min / (max - min);
	double scale = range / (max - min);
	bp.convertTo(bp, CV_32F, alpha, beta);

	// Dilate the image, which on CUDA involves converting it to a single byte format
	// In order for this to work you have to scale by 255 and 1/255
	bpt.convertTo(bpt, CV_8U, kToSingleByte);
	m_DilationKernel->apply(bpt, tmp);
	bpt.convertTo(bpt, CV_32F, kToFloat);

	// subtract off initial bandpass from dilated, store in lm
	cv::cuda::subtract(bp, lm, lm);

	// exponentiate to exxagerate
	cv::cuda::exp(lm, lm);

	// threshold so that things close to 1 stay and all else goes (booleans)
	cv::cuda::threshold(lm, lm, 1 - kEPS, 1, cv::THRESH_BINARY);

	// Convert local max image to binary single byte image (boolean)
	// maybe I should just work in single byte after the dilation...
	lm.convertTo(D.d_ParticleImg, CV_8U);
}