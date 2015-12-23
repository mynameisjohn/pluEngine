#include "CenterFind.h"

#include <algorithm>

ParticleStack::ParticleStack() :
m_uParticleCount(0),
m_uLastSliceIdx(0),
m_fMaxPeak(0.f) {
}

ParticleStack::ParticleStack(Particle first, uint32_t sliceIdx):
ParticleStack() {
	// Just invoke the function below
	AddParticle(first, sliceIdx);
}

uint32_t ParticleStack::AddParticle(Particle p, uint32_t sliceIdx) {
	// Update max peak and last slice idx
	m_fMaxPeak = std::max(m_fMaxPeak, p.i);
	m_uLastSliceIdx = std::max(m_uLastSliceIdx, sliceIdx);

	// Add particle to list
	m_liContributingParticles.push_back(p);

	// pre inc, return new count
	return ++m_uParticleCount;
}

// Get functions
uint32_t ParticleStack::GetParticleCount() const {
	return m_uParticleCount;
}

float ParticleStack::GetPeak() const {
	return m_fMaxPeak;
}

uint32_t ParticleStack::GetLastSliceIdx() const {
	return m_uLastSliceIdx;
}

Particle ParticleStack::GetLastParticleAdded() const {
	return m_liContributingParticles.back();
}

// Compute average particle (in 2D)
Particle ParticleStack::GetRefinedParticle() const {
	Particle ret{ 0 };
	float denom = 1.f / float(m_uParticleCount);
	for (auto& p : m_liContributingParticles) {
		ret.x += p.x * denom;
		ret.y += p.y * denom;
	}

	// Give it the max peak (supposed to be the center)
	ret.i = m_fMaxPeak;

	// Need peak idx, I think
	ret.z = float(m_uLastSliceIdx); 

	return ret;
}

// comparison operator impl
bool ParticleStack::comp::operator()(const ParticleStack& a, const ParticleStack& b) {
	return a.GetLastSliceIdx() < b.GetLastSliceIdx();
}