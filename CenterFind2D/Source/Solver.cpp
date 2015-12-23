#include "CenterFind.h"

#include <set>

Solver::Solver() :
m_uMaskRadius(0),
m_uFeatureRadius(0),
m_uMaxStackCount(0),
m_uNeighborRadius(0) {
}

Solver::Solver(uint32_t mR, uint32_t fR, uint32_t minSC, uint32_t maxSC, uint32_t nR) :
m_uMaskRadius(mR),
m_uFeatureRadius(fR),
m_uMinStackCount(minSC),
m_uMaxStackCount(maxSC),
m_uNeighborRadius(nR) {
	// Neighbor region diameter
	int diameter = 2 * m_uMaskRadius + 1;

	// Make host mats
	cv::Mat h_Circ(cv::Size(diameter, diameter), CV_32F, 0.f);
	cv::Mat h_RX = h_Circ;
	cv::Mat h_RY = h_Circ;
	cv::Mat h_R2 = h_Circ;

	// set up circle mask
	cv::circle(h_Circ, cv::Point(m_uMaskRadius, m_uMaskRadius), m_uMaskRadius, 1.f, -1);

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
	cv::threshold(h_R2, h_R2, pow((double)m_uMaskRadius, 2), 1, cv::THRESH_TOZERO_INV);
	cv::multiply(h_RX, h_Circ, h_RX);
	cv::multiply(h_RY, h_Circ, h_RY);

	// Copy to UMats
	h_Circ.copyTo(m_CircleMask);
	h_RX.copyTo(m_RadXKernel);
	h_RY.copyTo(m_RadYKernel);
	h_R2.copyTo(m_RadSqKernel);
}

uint32_t Solver::FindParticles(Datum& D) {
	// We're just doing this on the host for now
	// so download things here
	cv::Mat h_Input, h_LocalMax, h_ParticleImg;
	D.d_InputImg.download(h_Input);
	//D.d_LocalMaxImg.download(h_LocalMax);
	D.d_ParticleImg.download(h_ParticleImg);

	// Construct an Area of Interest only containing pixels
	// far enough in the image to count (img.width - 2*border == AOI.width)
	int border = m_uFeatureRadius;
	cv::Rect AOI({ border, border }, h_Input.size() - cv::Size(border, border));

	// We insert new particle stacks in order, to avoid sorting often
	std::vector <ParticleStack> foundParticles;

	// Precompute this, neighbor radius squared
	const float fNeighborRsq = powf((float)m_uNeighborRadius, 2);
	// Loop through every pixel of particle image inside AOI, stopping at nonzero values
	uint8_t * particleImgPtr = h_ParticleImg.ptr<uint8_t>();
	for (int xIdx = AOI.x; xIdx < AOI.width; xIdx++) {
		for (int yIdx = AOI.y; yIdx < AOI.height; yIdx++) {
			// Compute image index
			int idx = xIdx * AOI.width + yIdx;
			// If this value isn't zero, scan for a particle
			if (particleImgPtr[idx] != 0) {
				// Extract a region around the pixel(xIdx, yIdx) based on mask radius
				int mask = m_uMaskRadius;
				int diameter = m_uMaskRadius * 2 + 1;
				cv::Rect extract(xIdx - mask, yIdx - mask, diameter, diameter);
				cv::Mat e_Square = h_Input(extract);

				// multiply the extracted region by our circle mat
				cv::Mat product;
				cv::multiply(e_Square, m_CircleMask, product);

				// The sum corresponds to the mass of the particle at i,j
				float total_mass = cv::sum(product)[0];

				// If we have a particle, given that criteria
				if (total_mass > 0.f) {
					// Lambda to get x, y, r2 offset using Statistics kernels
					auto getOffset = [&e_Square, total_mass](cv::Mat& K) {
						cv::Mat product;
						cv::multiply(e_Square, K, product);
						float sum = cv::sum(product)[0];
						return sum / total_mass;
					};
					float x_offset = getOffset(m_RadXKernel) - 1;// -(mask + 1);
					float y_offset = getOffset(m_RadYKernel) - 1;// -(mask + 1);
					float r2_val = getOffset(m_RadSqKernel);

					// offset + index
					float x_val = x_offset + float(xIdx);
					float y_val = y_offset + float(yIdx);

					// We are only interested in linking up with particles
					// from the previous slice, so look in that range
					bool matchFound(false);
					int nParticlesSeached(0);
					for (auto rIt = m_vFoundParticles.rbegin(); rIt != m_vFoundParticles.rend(); ++rIt) {
						// Break if we're in a new slice Idx (slices are sorted by slice idx)
						if (rIt->GetLastSliceIdx() < (D.sliceIdx - 1))
							break;

						nParticlesSeached++;

						// Make a reference
						ParticleStack& pStack = *rIt;

						// We don't want too many stacks contributing to one particle
						if (pStack.GetParticleCount() < m_uMaxStackCount) {
							// We want the contributing particle in the previous slice
							// (what if it doesn't exist? I'm just going to ask for the top)
							// See if it's reasonably close to the last particle added
							// Another alternative here is to keep a running average
							// of the particle location in a stack (in xy) and check
							// the distance from that. Not sure what's the better option
							Particle prev = pStack.GetLastParticleAdded();
							float dX = prev.x - x_val;
							float dY = prev.y - y_val;
							float r2 = powf(dX, 2) + powf(dY, 2);

							// If it's close and the intensity is decreasing (?)
							if (r2 < fNeighborRsq /*&& prev.i > total_mass*/) {
								// make match and break

								// Add particle to stack, move this stack
								// to the "to be added" container so it doesn't
								// get picked up again or stagger the vector
								Particle p = { x_val, y_val, total_mass };
								pStack.AddParticle(p, D.sliceIdx);

								foundParticles.push_back(pStack);
								m_vFoundParticles.erase(std::next(rIt).base());
								matchFound = true;
								break;
							}
						}
					}

					// If no match was found, make a new particle
					if (matchFound == false) {
						Particle p = { x_val, y_val, total_mass };
						foundParticles.emplace_back(p, D.sliceIdx);
					}

					//// Construct particle, either add to existing stack or make new stack
					//Particle p = { x_val, y_val, total_mass };
					//if (matchStack) {
					//	matchStack->AddParticle(p, D.sliceIdx);
					//}
					//	
					//else 
					//	foundParticles.emplace_back(p, D.sliceIdx);
				}
			}
		}
	}

	// Insert all newly found particles

	//std::copy(foundParticles.begin(), foundParticles.end(), std::back_inserter(m_vFoundParticles));
	m_vFoundParticles.insert(m_vFoundParticles.end(), foundParticles.begin(), foundParticles.end());

	// Return count of newly found particles
	return foundParticles.size();
}

std::vector<Particle> Solver::GetFoundParticles() const{
	std::vector<Particle> ret(m_vFoundParticles.size());
	std::transform(m_vFoundParticles.begin(), m_vFoundParticles.end(), ret.begin(), 
				   [](const ParticleStack& pS) {return pS.GetRefinedParticle(); });
	return ret;
}