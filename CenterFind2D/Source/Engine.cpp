#include "CenterFind.h"
#include "FnPtrHelper.h"

#include <opencv2/cudaarithm.hpp>

#include <map>

void getUserParams(Datum D, BandPass * pEngineBP, LocalMax * pEngineLM, Solver * pEngineSolver);

Engine::Engine() {
}

bool Engine::Init(std::string stackPath, int startOfStack, int endOfStack) {
	// Attempt to open multibitmap
	FIMULTIBITMAP * FI_Input = FreeImage_OpenMultiBitmap(FIF_TIFF, stackPath.c_str(), 0, 1, 1, TIFF_DEFAULT);
	if (FI_Input == nullptr)
		return false;

	// Read in images, create data
	int i(0);
	for (int j = startOfStack; j < endOfStack; j++) {
		FIBITMAP * image = FreeImage_LockPage(FI_Input, j - 1);
		m_vData.emplace_back(image, i++);
	}

	// Dummy for now
	m_ParticleSolver = Solver(3, 6, 3, 5, 500);

	// Close multibitmap
	FreeImage_CloseMultiBitmap(FI_Input);
}

int Engine::Execute() {
	// Return if no data
	if (m_vData.empty())
		return -1;

	// This lets the user set DSP params
	getUserParams(m_vData.front(), &m_fnBandPass, &m_fnLocalMax, &m_ParticleSolver);

	// Run Centerfind algorithm on remaining images
	for (auto& D : m_vData) {
		m_fnBandPass(D);
		m_fnLocalMax(D);
		m_ParticleSolver.FindParticles(D);
	}

	auto shit = m_ParticleSolver.GetFoundParticles();
}

// This intentionally copies the Datum object, since we kind of modify it
void getUserParams(Datum D, BandPass * pEngineBP, LocalMax * pEngineLM, Solver * pEngineSolver) {
	// Window name
	std::string windowName = "PLuTARC CenterFind";

	// Trackbar Names
	std::string gaussRadiusTBName = "Gaussian Radius";
	std::string hwhmTBName = "Half-Width at Half-Maximum ";
	std::string dilationRadiusTBName = "Dilation Radius";
	std::string particleThreshTBName = "Particle Intensity Threshold";

	// We need pointers to these ints
	std::map<std::string, int> mapParamValues = {
		{ gaussRadiusTBName, 6 },	// These are the
		{ hwhmTBName, 4 },			// default values
		{ dilationRadiusTBName, 3 },// specified in the
		{ particleThreshTBName, 5 } // PLuTARC_testbed
	};

	// Trackbar callback, implemented below
	std::function<void(int, void *)> trackBarCallback = [&](int pos, void * priv) {
		// Construct operators based on current trackbar values
		BandPass fnBandPass(mapParamValues[gaussRadiusTBName], mapParamValues[hwhmTBName]);
		LocalMax fnLocalMax(mapParamValues[dilationRadiusTBName], mapParamValues[particleThreshTBName]);

		// Generate processed images
		fnBandPass(D);
		fnLocalMax(D);

		// returns formatted images for display
		auto makeDisplayImage = [](GpuMat& in) {
			GpuMat out = in;
			RemapImage(out, 0, 1);
			return out;
		};

		// Create larger display image (4 images, corner to corner)
		cv::Size dataSize = D.d_InputImg.size();
		cv::Size dispSize = dataSize;
		dispSize *= 2;	// Multiply by two in x and y
		GpuMat displayMat(dispSize, CV_32F, 0.f);

		// Display regions
		cv::Rect topLeft({ 0, 0 }, dataSize);
		cv::Rect topRight(cv::Rect({ dataSize.width, 0 }, dataSize));
		cv::Rect bottomLeft({ 0, dataSize.height }, dataSize);
		cv::Rect bottomRight({ dataSize.width, dataSize.height }, dataSize);

		// Copy all images to display image in correct place
		D.d_InputImg.copyTo(displayMat(topLeft));
		makeDisplayImage(D.d_FilteredImg).copyTo(displayMat(topRight));
		makeDisplayImage(D.d_DilateImg).copyTo(displayMat(bottomLeft));
		makeDisplayImage(D.d_LocalMaxImg).copyTo(displayMat(bottomRight));

		// Show new image
		cv::resizeWindow(windowName, dispSize.width, dispSize.height);
		cv::imshow(windowName, displayMat);
	};

	// Create window, just show input first
	cv::namedWindow(windowName, cv::WINDOW_OPENGL);
	
	// Create trackbars
	auto createTrackBar = [&mapParamValues, windowName, &trackBarCallback](std::string tbName, int maxVal) {
		auto it = mapParamValues.find(tbName);
		if (it != mapParamValues.end()) {
			cv::createTrackbar(tbName, windowName, &mapParamValues[tbName], maxVal, get_fn_ptr<0>(trackBarCallback));
		}
	};
	createTrackBar(gaussRadiusTBName, 15);
	createTrackBar(hwhmTBName, 15);
	createTrackBar(dilationRadiusTBName, 15);
	createTrackBar(particleThreshTBName, 15);

	// Call the callback on our own, just to pump things and show the images
	trackBarCallback(0, nullptr);

	// Wait while user sets things until they press a key (any key?)
	cv::waitKey();

	// Destroy window
	cv::destroyWindow(windowName);

	// Fill in pointers with new items
	*pEngineBP = BandPass(mapParamValues[gaussRadiusTBName], mapParamValues[hwhmTBName]);
	*pEngineLM = LocalMax(mapParamValues[dilationRadiusTBName], mapParamValues[particleThreshTBName]);
	*pEngineSolver = Solver(3, mapParamValues[gaussRadiusTBName], 3, 5, 5);
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