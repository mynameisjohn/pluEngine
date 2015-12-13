#include "CenterFind.h"

using namespace Centerfind;

Solver::Solver() :
m_uMaskRadius(0),
m_uFeatureRadius(0),
m_uMaxStackCount(0),
m_fNeighborRadius(0.f)
{}

Solver::Solver(int mask_radius, int feature_radius, int max_stack_count, float neighbor_radius) :
m_uMaskRadius(mask_radius),
m_uFeatureRadius(feature_radius),
m_uMaxStackCount(max_stack_count),
m_fNeighborRadius(neighbor_radius)
{}

Solver::FindParticles(Datum& D) {
	// NYI
}

std::vector<Particle> Solver::GetFoundParticles() {
	// NYI
}

Engine::Engine() {
	// This function now has to create windows and manage user input
}