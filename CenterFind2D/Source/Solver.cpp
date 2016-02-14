#include "CenterFind.h"
#include <iterator>
#include <algorithm>
#include <set>

// x, y, i default to -1
Particle::Particle(float x, float y, float i, int idx) :
z(idx),
peakIntensity(i),
nContributingParticles(1),
lastContributingsliceIdx(idx),
pState(Particle::State::NO_MATCH)
{
	this->x = x;
	this->y = y;
	this->i = i;
}

Solver::Solver() :
m_uMaskRadius(0),
m_uFeatureRadius(0),
m_uMaxStackCount(0),
m_uNeighborRadius(0) {
}

Solver::Solver(uint32_t mR, uint32_t fR, uint32_t minSC, uint32_t maxSC, uint32_t nR) :
m_uMaskRadius(mR),
m_nMaxLevel(3),
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

int pixelToGridIdx(float x, float y, int N, int m) {
	const int cellSize = N >> m;
	const int cellCount = N / cellSize;

	int cellX = x / cellSize;
	int cellY = y / cellSize;

	int cellIdx = cellX + cellCount * cellY;
	return cellIdx;
}

std::pair<int, int> gridIdxtoPixel(int idx, int N, int m) {
	const int cellSize = N >> m;
	const int cellCount = N / cellSize;

	int cellX = idx % cellCount;
	int cellY = idx / cellCount;

	return{ cellSize * cellX, cellSize * cellY };
}

int pixelToGridIdx(Particle p, int N, int m) {
	return pixelToGridIdx(p.x, p.y, N, m);
}

uint32_t Solver::FindParticles(Datum& D) {
	const int N = D.d_LocalMaxImg.rows;
	const int m = m_nMaxLevel;
	const int cellSize = N >> m;
	const int cellCount = N / cellSize;

	std::vector<Particle> vNewParticles, vUnmatchedParticles, vSortedParticleVec;

	// Cull the herd a bit
	int curSliceIdx = D.sliceIdx;
	m_vPrevParticles.erase(std::remove_if(m_vPrevParticles.begin(), m_vPrevParticles.end(), [curSliceIdx](const Particle& p) {
		// if the particle is far away and not in a severed state, bail (is this ok?)
		return (curSliceIdx - p.lastContributingsliceIdx > 2 && p.pState != Particle::State::SEVER);
	}), m_vPrevParticles.end());

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

	if (m_GridCells.empty()) {
		m_GridCells.resize(cellCount * cellCount);
	}
	memset(m_GridCells.data(), -1, sizeof(Cell) * m_GridCells.size());

	// set up ranges in cells given previous image
	// Find grid cell ranges
	// Thrust would make this cuter
	if (m_vPrevParticles.empty() == false) {
		std::vector<int> particleIndices(m_vPrevParticles.size());
		std::transform(m_vPrevParticles.begin(), m_vPrevParticles.end(), particleIndices.begin(),
			[N, m](const Particle& p) {return pixelToGridIdx(p, N, m); });

		//for (auto i : particleIndices)
		//	std::cout << i << std::endl;

		for (auto cellIt = m_GridCells.begin(); cellIt != m_GridCells.end(); ++cellIt) {
			int idx = std::distance(m_GridCells.begin(), cellIt);
			auto lowerIt = std::lower_bound(particleIndices.begin(), particleIndices.end(), idx);;
			auto upperIt = std::upper_bound(particleIndices.begin(), particleIndices.end(), idx);;
			cellIt->lower = std::distance(particleIndices.begin(), lowerIt);
			cellIt->upper = std::distance(particleIndices.begin(), upperIt);
		}
	}

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

					vNewParticles.emplace_back(x_val, y_val, total_mass, D.sliceIdx);
				}
			}
		}
	}

	for (auto& newParticle : vNewParticles) {
		// Set this particle's slice idx to the current slice idx
		newParticle.lastContributingsliceIdx = D.sliceIdx;
		newParticle.z = (float)D.sliceIdx;

		// Find this particle's cell
		int cellIdx = pixelToGridIdx(newParticle.x, newParticle.y, N, m);
		const Cell& cell = m_GridCells[cellIdx];

		// Not ideal, but who cares
		std::list<const Cell const *> cellsToSearch = { &cell };

		// Check left if we aren't on the left edge
		int cellX = cellIdx % cellCount;
		if (cellX != 0) {
			float left = cellX * cellSize;
			if (newParticle.x - m_uNeighborRadius < left)
				cellsToSearch.push_back(&m_GridCells[cellIdx - 1]);
		}

		// Check right if we aren't on the right edge
		if (cellX != cellCount - 1) {
			float right = (cellX + 1) * cellSize;
			if (newParticle.x + float(m_uNeighborRadius) > right)
				cellsToSearch.push_back(&m_GridCells[cellIdx + 1]);
		}

		// Check below if we aren't at the bottom
		int cellY = cellIdx / cellCount;
		if (cellY != 0) {
			float bottom = cellY * cellSize;
			if (newParticle.y - m_uNeighborRadius < bottom)
				cellsToSearch.push_back(&m_GridCells[cellIdx - cellCount]);
		}

		// Check above if we aren't on top
		if (cellY != cellCount - 1) {
			float top = (cellY + 1) * cellSize;
			if (newParticle.y + m_uNeighborRadius > top)
				cellsToSearch.push_back(&m_GridCells[cellIdx + cellCount]);
		}

		Particle * pBestMatch = nullptr;
		for (auto& cell : cellsToSearch) {
			for (int i = cell->lower; i != cell->upper; i++) {
				// Ref to potential match
				Particle& oldParticle = m_vPrevParticles[i];

				// In the original code only the previous slice was scanned, 
				// so try and uphold that I guess. A distance param is fine too (?)
				if (D.sliceIdx - oldParticle.lastContributingsliceIdx != 1)
					continue;

				// We can't have too many particles contributing to the same particle stack
				if (oldParticle.nContributingParticles > m_uMaxStackCount)
					continue;

				// If the particle stack has been severed, we don't care
				if (oldParticle.pState == Particle::State::SEVER)
					continue;

				// See if the particle is within our range
				float dX = oldParticle.x - newParticle.x;
				float dY = oldParticle.y - newParticle.y;
				float distSq = pow(dX, 2) + pow(dY, 2);

				if (distSq < float(m_uNeighborRadius * m_uNeighborRadius)) {
					// If there already was a match, see if this one is better
					if (pBestMatch) {
						// Find the old distance
						dX = pBestMatch->x - newParticle.x;
						dY = pBestMatch->y - newParticle.y;

						// If this one is closer, assign it as the match
						if (pow(dX, 2) + pow(dY, 2) > distSq)
							pBestMatch = &oldParticle;
					}
					else pBestMatch = &oldParticle;
				}
			}
		}

		// If we found a match, handle the intensity state logic
		if (pBestMatch != nullptr) {
			// We don't want to particles from the same slice contributing to a previous particle
			// Note that in my tests this never happens, so I'll probably nix it on the GPU
			if (pBestMatch->lastContributingsliceIdx == D.sliceIdx)
				pBestMatch = nullptr;
			// Otherwise handle intensity state logic
			else {
				switch (pBestMatch->pState) {
				case Particle::State::NO_MATCH:
					// Shouldn't ever get no match, but assign the state and fall through
					pBestMatch->pState = Particle::State::INCREASING;
				case Particle::State::INCREASING:
					// If we're increasing, see if the new guy prompts a decrease
					// Should we check to see if more than one particle has contributed?
					if (pBestMatch->i > newParticle.i)
						pBestMatch->pState = Particle::State::DECREASING;
					// Otherwise see if we should update the peak intensity and z position
					else if (newParticle.i > pBestMatch->peakIntensity) {
						pBestMatch->peakIntensity = newParticle.i;
						pBestMatch->z = (float)D.sliceIdx;
					}
					break;
				case Particle::State::DECREASING:
					// In this case, if it's still decreasing then fall through
					if (pBestMatch->i > newParticle.i)
						break;
					// If we're severing, assing the state and fall through
					pBestMatch->pState = Particle::State::SEVER;
				case Particle::State::SEVER:
					// Continue here (could catch this earlier)
					pBestMatch = nullptr;
				}
			}

			// If we didn't sever and null out above
			if (pBestMatch != nullptr) {
				// It's a match, bump the particle count and compute an averaged position (?)
				pBestMatch->nContributingParticles++;
				pBestMatch->lastContributingsliceIdx = D.sliceIdx;

				// If I don't do the average pos thing, 

				// I don't know about the averaged position thing
				pBestMatch->x = 0.5f * (pBestMatch->x + newParticle.x);
				pBestMatch->y = 0.5f * (pBestMatch->y + newParticle.y);
			}
		}

		// If this is still null, there isn't a goood matching particle, so make a new one
		if (pBestMatch == nullptr)
			vUnmatchedParticles.push_back(newParticle);
	}

	// tack on unmatched particles and sort (will be more complicated in thrust
	m_vPrevParticles.insert(m_vPrevParticles.end(), vUnmatchedParticles.begin(), vUnmatchedParticles.end());
	std::sort(m_vPrevParticles.begin(), m_vPrevParticles.end(), 
		[N, m](const Particle& a, const Particle& b) {
		return pixelToGridIdx(a, N, m) < pixelToGridIdx(b, N, m);
	});


	
	// Because position is being updated, I'm doing the sort above instead of the inplace merge
	//std::sort(vUnmatchedParticles.begin(), vUnmatchedParticles.end(),
	//	[N, m](const Particle& a, const Particle& b) {
	//	return pixelToGridIdx(a, N, m) < pixelToGridIdx(b, N, m);
	//});

	//auto sortIt = m_vPrevParticles.insert(m_vPrevParticles.end(), vUnmatchedParticles.begin(), vUnmatchedParticles.end());
	//std::inplace_merge(m_vPrevParticles.begin(), sortIt, m_vPrevParticles.end(), [N, m](const Particle& a, const Particle& b) {
	//	return pixelToGridIdx(a, N, m) < pixelToGridIdx(b, N, m);
	//});



	std::cout << "Slice Idx:\t" << D.sliceIdx << "\tNew Particles:\t" << vNewParticles.size() << "\tUnmatched Particles:\t" << vUnmatchedParticles.size() << "\tFound Particles:\t" << m_vPrevParticles.size() << std::endl;

	return m_vPrevParticles.size();
}

std::vector<Particle> Solver::GetFoundParticles() const{

	int nParticles = std::count_if(m_vPrevParticles.begin(), m_vPrevParticles.end(), [](const Particle& p) {
		return p.pState == Particle::State::SEVER && p.nContributingParticles > 2;
	});
	std::cout << "Final particle count: " << nParticles << std::endl;

	std::vector<Particle> ret(m_vFoundParticles.size());
	std::transform(m_vFoundParticles.begin(), m_vFoundParticles.end(), ret.begin(), 
				   [](const ParticleStack& pS) {return pS.GetRefinedParticle(); });
	return ret;
}