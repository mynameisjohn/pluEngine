#pragma once

#include "CenterFind.h"

#include <thrust/device_vector.h>

struct Cell
{
	int lower;
	int upper;
};

struct Particle
{
	enum class State
	{
		NO_MATCH = 0,
		INCREASING,
		DECREASING,
		SEVER
	};
	float x;
	float y;
	float z;
	float i;
	float peakIntensity;
	int nContributingParticles;
	int lastContributingsliceIdx;
	State pState;

	__host__ __device__
		Particle( float x = -1.f, float y = -1.f, float i = -1.f, int idx = -1 );
};

class Solver
{
	// Useful typedefs of mine
	using UcharVec = thrust::device_vector < unsigned char >;
	using UcharPtr = thrust::device_ptr < unsigned char > ;

	using IntVec = thrust::device_vector < int > ;
	using IntPtr = thrust::device_ptr < int > ;

	using FloatVec = thrust::device_vector < float >;
	using Floatptr = thrust::device_ptr < float > ;

	using ParticleVec = thrust::device_vector < Particle >;
	using ParticlePtrVec = thrust::device_vector < Particle * >;

	using Img = GpuMat;

	/// I use these for host debugging
	//using UcharVec = thrust::host_vector < unsigned char >;
	//using UcharPtr = unsigned char *;
	//using IntVec = thrust::host_vector < int >;
	//using IntPtr = int *;
	//using FloatVec = thrust::host_vector < float >;
	//using Floatptr = float *;
	//using ParticleVec = thrust::host_vector < Particle >;
	//using ParticlePtrVec = thrust::host_vector < Particle * >;
	//using Img = cv::Mat;

public:
	Solver();
	Solver( int mR, int fR, int minSC, int maxSC, int nR );
	int FindParticles( Datum& D );
	std::vector<Particle> GetFoundParticles() const;
private:
	int m_uMaskRadius;				// The radius of our particle mask kernels
	int m_uFeatureRadius;			// The radius within the image we'd like to consier
	int m_uMinSliceCount;			// The minimum # of slices we require to contribute to a particle
	int m_uMaxSliceCount;			// The maximum # of slices we allow to contribute to a particle
	int m_uNeighborRadius;			// The radius in which we search for new particles
	int m_uMaxLevel;				// The subdivision level we use to spatially partition previous particles

	Img m_dCircleMask;					// The circle mask, just a circle of radius m_uMaskRadius, each value is 1
	Img m_dRadXKernel;					// The x mask, used to calculate an offset to the x coordinate
	Img m_dRadYKernel;					// The y mask, used to calculate an offset to the y coordinate
	Img m_dRadSqKernel;					// The r2 mask, used to calculate some value that I don't really understand

	size_t m_uCurPrevParticleCount;		// The current tally of previous particles to search through
	ParticleVec m_dPrevParticleVec;		// The vector of previously found particles

	IntVec m_dGridCellLowerBoundVec;	// The lower bound vector of particles to search
	IntVec m_dGridCellUpperBoundVec;	// The upper bound vector of particles to search

	// These are private functions that actually do the solving
private:
	size_t cullExistingParticles( int curSliceIdx );		// Remove found particles if they are deemed to be noise
	ParticleVec findNewParticles( UcharVec& d_ParticleImgVec, Floatptr pThreshImg, int N, int sliceIdx );
	void createGridCells( int N );
	ParticlePtrVec findParticleMatches( ParticleVec& d_NewParticleVec, int N, int sliceIdx );
	void updateMatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec, int sliceIdx );
	Solver::ParticleVec consolidateUnmatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec );
	void mergeUnmatchedParticles( ParticleVec& d_UnmatchedParticleVec, int N );
};