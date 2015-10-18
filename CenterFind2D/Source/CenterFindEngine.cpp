#include "CenterFindEngine.h"
#include <iostream>
#include <array>
#include <string>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Constants
const float kEPS(0.0000001f);

static void RecenterImage(cv::UMat& img, float m = 0.f, float M = 100.f) {
	double range = M - m;
	double min(1), max(2);
	cv::minMaxIdx(img, &min, &max);
	double scale = range / (max - min);
	cv::subtract(img, min, img);
	cv::multiply(img, scale, img);
}

static void ShowImage(cv::UMat& img) {
	cv::UMat disp = img;
	RecenterImage(disp, 0.f, 1.f);
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display Window", disp);
	cv::waitKey(0);
}

CenterFindEngine::Parameters::Parameters() :
	m_uFileNamePad(0),
	m_uStartOfStack(0),
	m_uEndOfStack(0),
	m_uStartFrame(0),
	m_uEndFrame(0),
	m_uFeatureRadius(0),
	m_uDilationRadius(0),
	m_uMaskRadius(0),
	m_fHWHMLength(0),
	m_fPctleThreshold(0)
{}

CenterFindEngine::Parameters::Parameters(std::array<std::string, 12> args){
	uint32_t idx(0);
	m_strInputStem = args[idx++];
	m_strOutputStem = args[idx++];
	m_strFileExt = ".tif";
	m_uFileNamePad = 4; // ?
	std::stringstream(args[idx++]) >> m_uStartOfStack;
	std::stringstream(args[idx++]) >> m_uEndOfStack;
	std::stringstream(args[idx++]) >> m_uStartFrame;
	std::stringstream(args[idx++]) >> m_uEndFrame;
	std::stringstream(args[idx++]) >> m_uFeatureRadius;
	std::stringstream(args[idx++]) >> m_uDilationRadius;
	std::stringstream(args[idx++]) >> m_uMaskRadius;
	std::stringstream(args[idx++]) >> m_fHWHMLength;
	std::stringstream(args[idx++]) >> m_fPctleThreshold;
	m_setOutputMode = { OutputMode::TEXT };
}

bool CenterFindEngine::Parameters::IsOutputModeOn(CenterFindEngine::OutputMode om) {
	return m_setOutputMode.find(om) != m_setOutputMode.end();
}

std::string CenterFindEngine::Parameters::GetFileName(uint32_t idx) {
	std::string num = std::to_string(idx);
	while (num.length() < m_uFileNamePad)
		num = std::string("0").append(num);
	return m_strInputStem + "_" + num + m_strFileExt;
}

CenterFindEngine::Data::Data(FIBITMAP * bmp) {
	cv::Mat image = cv::Mat::zeros(FreeImage_GetWidth(bmp), FreeImage_GetHeight(bmp), CV_8UC3);
	
	FreeImage_ConvertToRawBits(image.data, bmp, image.step, 24, 0xFF, 0xFF, 0xFF, true);
	
	cvtColor(image, image, CV_RGB2GRAY);
	
	image.convertTo(image, CV_32FC1);

	m_BypassedImg = cv::UMat(image.size(), image.type(), 0.f);
	m_ThresholdImg = cv::UMat(image.size(), image.type(), 0.f);
	m_LocalMaxImg = cv::UMat(image.size(), image.type(), 0.f);
	m_ParticleImg = cv::UMat(image.size(), image.type(), 0.f);
}

CenterFindEngine::BandPass::BandPass() :
	m_uGaussianRadius(0)
{}

CenterFindEngine::BandPass::BandPass(int radius, float hwhm) :
m_uGaussianRadius(radius)
{
	uint32_t diameter = 2 * m_uGaussianRadius + 1;

	// Create gaussian filter image
	cv::Mat h_Gaussian = cv::getGaussianKernel(diameter, (hwhm / 0.8325546) / 2, CV_32F);

	// Create circle image
	cv::Mat h_Circle = cv::Mat::zeros(cv::Size(diameter, diameter), CV_32F);
	cv::circle(h_Circle, cv::Size(m_uGaussianRadius, m_uGaussianRadius), m_uGaussianRadius, 1.f, -1);

	// Put in UMats
	h_Gaussian.copyTo(m_GaussKernel);
	h_Circle.copyTo(m_CircleMask);
}

void CenterFindEngine::BandPass::Execute(CenterFindEngine::Data& data) {
	// Make some references
	cv::UMat& in = data.m_InputImg;
	cv::UMat& bp = data.m_BypassedImg;

	// Make tmp buffer
	cv::UMat tmp(in.size(), in.type(), 0.f);

	// Apply Gaussian filter, store in out
	cv::sepFilter2D(in, bp, -1, m_GaussKernel, m_GaussKernel);

	// Apply circle mask, store in tmp
	cv::filter2D(in, tmp, -1, m_CircleMask);

	// scale tmp down
	float scale = 1 / (3 * pow(m_uGaussianRadius, 2));
	cv::divide(tmp, (3 * pow(m_uGaussianRadius, 2)), tmp);

	// subtract tmp from bandpass to get filtered output
	cv::subtract(bp, tmp, bp);

	// Any negative values become 0
	cv::threshold(bp, bp, 0, 1, cv::THRESH_TOZERO);
}

CenterFindEngine::LocalMax::LocalMax() :
	m_uDilationRadius(0),
	m_fPctleThreshold(0)
{}

CenterFindEngine::LocalMax::LocalMax(int radius, float pctl_thresh) :
m_uDilationRadius(radius),
m_fPctleThreshold(pctl_thresh)
{
	uint32_t diameter = 2 * m_uDilationRadius + 1;

	cv::Mat h_Dilation = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(diameter, diameter));

	h_Dilation.copyTo(m_DilationKernel);
}

void CenterFindEngine::LocalMax::Execute(CenterFindEngine::Data& data) {
	// Make some references
	cv::UMat& in = data.m_InputImg;
	cv::UMat& bp = data.m_BypassedImg;
	cv::UMat& bpThresh = data.m_ThresholdImg;
	cv::UMat& lm = data.m_LocalMaxImg;
	cv::UMat& particles = data.m_ParticleImg;

	// Tmp buffer
	cv::UMat tmp(in.size(), CV_8U, m_fPctleThreshold);

	// Set the threshold to be entirely base value
	bpThresh.setTo(m_fPctleThreshold);
	// recenter bandpassed
	RecenterImage(bp);
	// bpThresh is >= m_fPctleThreshold
	max(bp, m_fPctleThreshold, bpThresh);

	// Perform dilation, store in tmp
	cv::dilate(bp, tmp, m_DilationKernel);

	// subtract off initial bandpas from tmp, store in lm
	cv::subtract(bp, tmp, lm);

	// exponentiate to exxagerate
	cv::exp(lm, lm);

	// threshold so that things close to 1 stay and all else goes
	cv::threshold(lm, lm, 1 - kEPS, 1, cv::THRESH_BINARY);
}

CenterFindEngine::Statistics::Statistics():
	m_uMaskRadius(0),
	m_uFeatureRadius(0)
{}

CenterFindEngine::Statistics::Statistics(int mask_radius, int feature_radius) :
	m_uMaskRadius(mask_radius),
	m_uFeatureRadius(feature_radius)
{
	// circle diameter
	int diameter = 2 * m_uMaskRadius + 1;

	// Make host mats
	cv::Mat h_Circ(cv::Size(diameter, diameter), 0.f, CV_32F);
	cv::Mat h_RX = h_Circ;
	cv::Mat h_RY = h_Circ;
	cv::Mat h_R2 = h_Circ;

	// set up circle
	cv::circle(h_Circ, cv::Point(mask_radius, mask_radius), mask_radius, 1.f, -1);

	// set up Rx and part of r2
	for (int i = 0; i < diameter; i++) {
		for (int j = 0; j < diameter; j++) {
			h_RX.at<float>(i, j) = float(j + 1);
			h_R2.at<float>(i, j) += float(pow(j - m_uMaskRadius, 2));
		}
	}

	// set up Ry and the rest of r2
	for (int i = 0; i < diameter; i++) {
		for (int j = 0; j < diameter; j++) {
			h_RY.at<float>(i, j) = float(i + 1);
			h_R2.at<float>(i, j) += float(pow(i - m_uMaskRadius, 2));
		}
	}

	// I forget what these do...
	cv::threshold(h_R2, h_R2, pow(mask_radius, 2), 1, cv::THRESH_TOZERO_INV);
	cv::multiply(h_RX, h_Circ, h_RX);
	cv::multiply(h_RY, h_Circ, h_RY);

	// Copy to UMats
	h_Circ.copyTo(m_CircleMask);
	h_RX.copyTo(m_RadXKernel);
	h_RY.copyTo(m_RadYKernel);
	h_R2.copyTo(m_RadSqKernel);
}

CenterFindEngine::PMetricsVec CenterFindEngine::Statistics::GetMetrics(CenterFindEngine::Data& data) {
	cv::UMat& input = data.m_InputImg;
	cv::UMat& lm = data.m_LocalMaxImg;
	
	PMetricsVec ret;

	int border = m_uFeatureRadius;
	cv::Size sz = input.size();
	cv::Rect AOI(border, border, sz.width - 2 * border, sz.height - 2 * border);
	int diameter = m_uMaskRadius * 2 + 1;

	// Until I have a kernel...
	cv::Mat h_ParticleImg;
	data.m_ParticleImg.copyTo(h_ParticleImg);

	// TODO this has to be a kernel, so I'll lose most of the OCV niceties
	for (int i = 0; i < sz.width; i++) {
		for (int j = 0; j < sz.height; j++) {
			int idx = i*sz.width + j;
			char * ptr = h_ParticleImg.ptr<char>();
			if (ptr[idx] != 0 && AOI.contains(cv::Point(i, j))) {
				// Extract a region around i,j based on mask radius
				int mask = m_uMaskRadius;
				cv::Rect extract(i - mask, j - mask, diameter, diameter);
				cv::UMat e_Square = data.m_InputImg(extract);

				// multiply the extracted region by our circle mat
				cv::UMat product;
				cv::multiply(e_Square, m_CircleMask, product);

				// The sum corresponds to the mass of the particle at i,j
				float total_mass = cv::sum(product)[0];

				// If we have a particle, given that criteria
				if (total_mass > 0.f) {
					// Create the metrics struct
					CenterFindEngine::ParticleMetrics pMet{ 0 };
					pMet.idx = float(idx);
					pMet.mass = total_mass;

					// Sum local bool region 
					cv::UMat m_Square = data.m_ParticleImg(extract);
					pMet.multiplicity = cv::sum(m_Square)[0];

					// Lambda to get x, y, r2 offset using Statistics kernels
					auto getOffset = [&product, &e_Square, total_mass](cv::UMat& K) {
						cv::multiply(e_Square, K, product);
						return ((cv::sum(product)[0]) / total_mass);
					};					
					pMet.x_offset = getOffset(m_RadXKernel) - (mask + 1);
					pMet.y_offset = getOffset(m_RadYKernel) - (mask + 1);
					pMet.r2_val = getOffset(m_RadSqKernel);

					// offset + index
					pMet.x_val = pMet.x_offset + float(i);
					pMet.y_val = pMet.y_offset + float(j);

					// Store each found particle in our return vector
					ret.push_back(pMet);
				}
			}
		}
	}

	return ret;
}

CenterFindEngine::CenterFindEngine(const CenterFindEngine::Parameters params) :
m_Params(params)
{
	for (int i = m_Params.m_uStartOfStack; i < m_Params.m_uEndOfStack; i++) {
		std::string fileName = m_Params.GetFileName(i);
		FIMULTIBITMAP * FI_Input = FreeImage_OpenMultiBitmap(FIF_TIFF, fileName.c_str(), 0, 1, 1, TIFF_DEFAULT);
		for (int j = m_Params.m_uStartFrame; j < m_Params.m_uEndFrame; j++)
			m_Images.emplace_back(FreeImage_LockPage(FI_Input, j - 1));
		FreeImage_CloseMultiBitmap(FI_Input, TIFF_DEFAULT);
	}
}

std::deque<CenterFindEngine::PMetricsVec> CenterFindEngine::Execute() {
	std::deque<PMetricsVec> ret;

	for (auto& data : m_Images) {
		m_fnBandPass.Execute(data);
		m_fnLocalMax.Execute(data);

		ret.emplace_back(m_fnStatistics.GetMetrics(data));
	}

	return ret;
}

//
//#include <opencv2/gpu/gpu.hpp>
//
//void CenterFindEngine::showImage(Mat& img){
//	Mat x = img;
//
//	//RecenterImage(x, 1);
//	double min(1), max(2);
//	minMaxLoc(x, &min, &max);
//	x = (x - min) / (max - min);
//
//	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//	imshow("Display window", x);                   // Show our image inside it.
//	waitKey(0);
//}
//
//void CenterFindEngine::RecenterImage(Image& img, double range){
//	double min(1), max(2);
//	minMaxLoc(img, &min, &max);
//	subtract(img, min, img);
//
//	//cout << min << "\n" << max << endl;
//#ifdef OCL_OCV
//	multiply(range / (max - min), img, img);
//#elif defined CU_OCV
//	multiply(img,range / (max - min),img);
//#else
//	img *= (range / (max - min));
//#endif
//}
//
//CenterFindEngine::Parameters::Parameters(string params[12]){
//	int idx(0);
//
//	infile_stem = params[idx++];
//	outfile_stem = params[idx++];
//	file_extension = ".tif";
//
//	sstrm(params[idx++]) >> start_frameofstack;
//	sstrm(params[idx++]) >> end_frameofstack;
//	sstrm(params[idx++]) >> start_stack;
//	sstrm(params[idx++]) >> end_stack;
//
//	sstrm(params[idx++]) >> feature_radius;
//	sstrm(params[idx++]) >> hwhm_length;
//	sstrm(params[idx++]) >> dilation_radius;
//	sstrm(params[idx++]) >> mask_radius;
//	sstrm(params[idx++]) >> pctle_threshold;
//	sstrm(params[idx]) >> testmode;
//
//}
//
//CenterFindEngine::BandPassEngine::BandPassEngine(int radius, float h)
//	: m_Radius(radius){
//	int diameter = 2 * m_Radius + 1;
//
//#if defined OCL_OCV || defined CU_OCV
//	//Mat h_Gaussian = getGaussianKernel(diameter, (h / 0.8325546) / 2, CV_32F);
//	Mat h_Circle = Mat::zeros({ diameter, diameter }, CV_32F);
//	circle(h_Circle, { m_Radius, m_Radius }, m_Radius, 1.f, -1);
//
//	Gaussian = createGaussianFilter_GPU(CV_32F, { diameter, diameter }, (h / 0.8325546) / 2);
//	Circle = createLinearFilter_GPU(CV_32F, CV_32F, h_Circle);
//#else
//	Gaussian = createGaussianFilter(CV_32F, { diameter, diameter }, (h / 0.8325546) / 2);
//	//Gaussian = getGaussianKernel(diameter, (h / 0.8325546) / 2, CV_32F);
//	Mat tmp_Circle = Mat::zeros({ diameter, diameter }, CV_32F);
//	circle(tmp_Circle, { m_Radius, m_Radius }, m_Radius, 1.f, -1);
//	Circle = createLinearFilter(CV_32F, CV_32F, tmp_Circle);
//
//	//Gaussian = Image(h_Gaussian);
//	//Circle = Image(h_Circle);
//#endif
//
//}
//
//void CenterFindEngine::BandPassEngine::operator() (CenterFindData& img){
//	//sepFilter2D(img.in, img.bpass, -1, Gaussian, Gaussian);
//	//filter2D(img.in, img.tmp, -1, Circle);
//	Gaussian->apply(img.in, img.bpass);
//	Circle->apply(img.in,img.tmp);
//
//#ifdef OCL_OCV
//	multiply(1 / (3 * pow(m_Radius, 2)), img.tmp, img.tmp);
//#elif defined CU_OCV
//	divide(img.tmp, 3 * pow(m_Radius, 2), img.tmp);
//#else
//	img.tmp = img.tmp / (3 * pow(m_Radius, 2));
//#endif
//	subtract(img.bpass, img.tmp, img.bpass);
//	threshold(img.bpass, img.bpass, 0, 1, THRESH_TOZERO);
//}
//
//CenterFindEngine::LocalMaxEngine::LocalMaxEngine(int radius, float pctl_thresh)
//	: m_Radius(radius), m_Pctl_Threshold(pctl_thresh){
//	int diameter = 2 * m_Radius + 1;
//
//	//Works with OCL?
//	Mat tmp_Dilation = getStructuringElement(MORPH_ELLIPSE, { diameter, diameter });
//#if defined OCL_OCV || defined CU_OCV
//	Dilation = createMorphologyFilter_GPU(MORPH_DILATE, CV_8U, tmp_Dilation);
//#else
//	Dilation = createMorphologyFilter(MORPH_DILATE, CV_32F, tmp_Dilation);
//#endif
//}
//
//void CenterFindEngine::LocalMaxEngine::operator()(CenterFindData& img){
	//const float epsilon(0.0000001f);

	////img.bpass_thresh = m_Pctl_Threshold;
	//img.bpass_thresh.setTo(m_Pctl_Threshold);
	//RecenterImage(img.bpass);
	//max(img.bpass, Threshold, img.bpass_thresh);



	
//Both the CUDA and OCL OpenCV docs say this is only compatible with 
//byte based image formats. I'm getting away with it on OCL somehow...
//#ifdef CU_OCV
//	multiply(img.bpass_thresh,255.f,img.bpass_thresh);
/*
	img.bpass_thresh.convertTo(img.bpass_thresh,CV_8U);
	img.tmp.convertTo(img.tmp,CV_8U);

	Dilation->apply(img.bpass_thresh, img.tmp);

	img.bpass_thresh.convertTo(img.bpass_thresh,CV_32F);
	img.tmp.convertTo(img.tmp,CV_32F);
*/
//	divide(img.bpass_thresh,255.f,img.bpass_thresh);
//	divide(img.tmp,255.f,img.tmp);
//#else
//	Dilation->apply(img.bpass_thresh, img.tmp);
//#endif
//	//dilate(img.bpass_thresh, img.tmp, Dilation);
//	subtract(img.bpass, img.tmp, img.local_max);
//	exp(img.local_max, img.local_max);
//	threshold(img.local_max, img.local_max, 1 - epsilon, 1, THRESH_BINARY);

	//img.local_max.convertTo(img.local_max, CV_8U);
	/*

		exp(img.local_max, img.local_max);
		double m(0), M(0);
		minMaxLoc(img.local_max, &m, &M);

		threshold(img.local_max, img.local_max, 1 - epsilon, 1, THRESH_BINARY);*/
//}
//
//CenterFindEngine::StatisticsEngine::StatisticsEngine(int mask_radius, int feature_radius)
//	: m_Mask_Radius(mask_radius), m_Feature_Radius(feature_radius){
//	int diameter = 2 * m_Mask_Radius + 1;
//
//	//cv::Mat_<float> circTmp = cv::Mat_<float>(diameter, diameter, 0.f),
//	//	rxTmp = cv::Mat_<float>(diameter, diameter, 0.f),
//	//	ryTmp = cv::Mat_<float>(diameter, diameter, 0.f),
//	//	r2Tmp = cv::Mat_<float>(diameter, diameter, 0.f);
//
//	Circle = rX = rY = r2 = cv::Mat_<float>(diameter, diameter, 0.f);
//
//	circle(Circle, { m_Mask_Radius, m_Mask_Radius }, m_Mask_Radius, 1.f, -1);
//
//	for (int i = 0; i < diameter; i++)
//		for (int j = 0; j < diameter; j++){
//			rX.at<float>(i, j) = float(j + 1);
//			r2.at<float>(i, j) += float(pow(j - m_Mask_Radius, 2));
//		}
//
//	for (int i = 0; i < diameter; i++)
//		for (int j = 0; j < diameter; j++){
//			rY.at<float>(i, j) = float(i + 1);
//			r2.at<float>(i, j) += float(pow(i - m_Mask_Radius, 2));
//		}
//
//	threshold(r2, r2, pow(mask_radius, 2), 1, THRESH_TOZERO_INV);
//	multiply(rX, Circle, rX);
//	multiply(rY, Circle, rY);
//
//	////Change the constructor?
//	//Circle = Image(circTmp);
//	//rX = Image(rxTmp);
//	//rY = Image(ryTmp);
//	//r2 = Image(r2Tmp);
//
////#ifdef CU_OCV
////	Circle.upload(circTmp);
////	rX.upload(rxTmp);
////	rY.upload(ryTmp);
////	r2.upload(r2Tmp);
////#else
////
////	//When OCL is on, these do the upload
////	Circle = circTmp;
////	rX = rxTmp;
////	rY = ryTmp;
////	r2 = r2Tmp;
////#endif
//}
//
//vector<CenterFindEngine::ParticleData> CenterFindEngine::StatisticsEngine::operator()(CenterFindData& img){
//	const float epsilon(0.0001f);
//
//	vector<ParticleData> ret;
//
//	int counter(0);
//	int border = m_Feature_Radius;
//	int minx = border, miny = border;
//	int maxx = img.cols() - minx;
//	int maxy = img.rows() - minx;
//	int diameter = m_Mask_Radius * 2 + 1;
//
////#if !(defined OCL_OCV || defined CU_OCV)
//	for (int i = 0; i < img.rows()*img.cols(); i++){
//		auto * ptr = img.particles.ptr<unsigned char>();
//		if (ptr[i]/*fabs(ptr[i] - 1) < epsilon*/){
//			int xval(i%img.cols());
//			int yval(floor((float)i / img.cols()));
//			if (xval > minx && xval < maxx && yval > miny && yval < maxy) {
//				int mask = m_Mask_Radius;
//				Rect extract(xval - mask, yval - mask, diameter, diameter);
//				Mat e_square(img.input(extract)), result;
//				multiply(e_square, Circle, result);
//				float total_mass(sum(result)[0]);
//
//				if (total_mass > 0) {
//					multiply(e_square, rX, result);
//					float x_offset = ((sum(result)[0]) / total_mass) - mask - 1;
//
//					multiply(e_square, rY, result);
//					float y_offset = (sum(result)[0] / total_mass) - mask - 1;
//
//					multiply(e_square, r2, result);
//					float r2_val = (sum(result)[0] / total_mass);
//
//					Mat m_square = img.particles(extract);
//					float multiplicity(sum(m_square)[0]);
//
//					ParticleData p = {
//						float(i),
//						xval + x_offset,
//						yval + y_offset,
//						x_offset,
//						y_offset,
//						total_mass,
//						r2_val,
//						multiplicity
//					};
//					ret.push_back(p);
//
//					counter++;
//				}
//			}
//		}
//	}
////#endif
////	cout << counter << endl;
//	return ret;
//}
//#include <chrono>
//#include <thread>
//
////Resource Aquisition Is Initialization?
//CenterFindEngine::CenterFindEngine(string params[12])
//	: m_Params(params),
//	m_BandPass(m_Params.feature_radius, m_Params.hwhm_length),
//	m_LocalMax(m_Params.dilation_radius, m_Params.pctle_threshold),
//	m_Statistics(m_Params.mask_radius, m_Params.feature_radius)
//{
//	auto TIFF_to_OCV = [](FIBITMAP * dib){
//		Mat image = Mat::zeros(FreeImage_GetWidth(dib), FreeImage_GetHeight(dib), CV_8UC3);
//		Image ret;
//
//		FreeImage_ConvertToRawBits(image.data, dib, image.step, 24, 0xFF, 0xFF, 0xFF, true);
//
//		cvtColor(image, image, CV_RGB2GRAY);
//
//		image.convertTo(image, CV_32FC1);
//
//		//ret = Image(image);
//
//		return image; //ret;
//	};
//	for (int i = m_Params.start_stack; i < m_Params.end_stack; i++){
//		string fileName = m_Params.getFileName(i).c_str();
//
//		FIMULTIBITMAP * FI_input =
//			FreeImage_OpenMultiBitmap(FIF_TIFF, m_Params.getFileName(i).c_str(),
//			FALSE, TRUE, TRUE, TIFF_DEFAULT);
//
//		for (int j = m_Params.start_frameofstack; j < m_Params.end_frameofstack; j++)
//			m_Images.emplace_back(TIFF_to_OCV(FreeImage_LockPage(FI_input, j - 1)));
//
//		FreeImage_CloseMultiBitmap(FI_input, TIFF_DEFAULT);
//	}
//
//
//	m_LocalMax.Threshold = Image({ m_Images.front().rows(), m_Images.front().cols() }, CV_32F, m_Params.pctle_threshold);
//
//	thread T([&](){
//		int nProcessed(0);
//		for (auto& img : m_Images){
//			unique_lock<mutex> lk(img.m);
//			img.cv.wait(lk, [&img]{return img.goodToGo; });
//			img.m_Data = m_Statistics(img);
//			lk.unlock();
//			//cout << "Statistics gathered from " << nProcessed++ << " images" << endl;
//		}
//	});
//
//	auto begin = chrono::steady_clock::now();
//	int nProcessed(0);
//	for (auto& img : m_Images){
//
//		Mat display = Mat(img.in);
////		img.in.download(display);
//		showImage(display);
//
//		RecenterImage(img.in);
//
//		m_BandPass(img);
//
//		display = Mat(img.bpass);/*.download(display);*/
//		showImage(display);
//
//		m_LocalMax(img);
////		img.local_max.download(display);
//		display = Mat(img.local_max);
//		showImage(display);
//
//		img.local_max.convertTo(img.local_max, CV_8U);
//#ifndef CU_OCV
//		img.particles = img.local_max;
//#else
//		img.local_max.download(img.particles);
//#endif
//		RecenterImage(img.bpass_thresh);
//		{
//			lock_guard<mutex> lk(img.m);
//			img.goodToGo = true;
//		}
//		img.cv.notify_one();
//		//cout << "Filters run on " << nProcessed++ << " images" << endl;
//		//m_Data.push_back(m_Statistics(img));
//	}
//	T.join();
//	auto end = chrono::steady_clock::now();
//
//	string msg;
//
//#ifdef OCL_OCV
//	msg = "OpenCL time took ";
//#elif defined CU_OCV
//	msg = "CUDA time took ";
//#else
//	msg = "Host time took ";
//#endif
//
//	cout << msg << chrono::duration<double, milli>(end-begin).count() << endl;
//}
