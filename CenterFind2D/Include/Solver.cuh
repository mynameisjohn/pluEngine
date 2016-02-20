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
public:

	// Make device pointers to the kernels used in particle solving and the localmax img
	using dUcharVec = thrust::device_vector < unsigned char >;
	using dUcharPtr = thrust::device_ptr < unsigned char > ;

	using dIntVec = thrust::device_vector < int > ;
	using dIntPtr = thrust::device_ptr < int > ;

	using dFloatVec = thrust::device_vector < float >;
	using dFloatptr = thrust::device_ptr < float > ;

	using dParticleVec = thrust::device_vector < Particle >;
	using dParticlePtrVec = thrust::device_vector < Particle * >;

	using dImg = GpuMat;

	//using dUcharVec = thrust::host_vector < unsigned char >;
	//using dUcharPtr = unsigned char *;

	//using dIntVec = thrust::host_vector < int >;
	//using dIntPtr = int *;

	//using dFloatVec = thrust::host_vector < float >;
	//using dFloatptr = float *;

	//using dParticleVec = thrust::host_vector < Particle >;
	//using dParticlePtrVec = thrust::host_vector < Particle * >;

	//using dImg = cv::Mat;

	Solver();
	Solver( uint32_t mR, uint32_t fR, uint32_t minSC, uint32_t maxSC, uint32_t nR );
	uint32_t FindParticles( Datum& D );
	std::vector<Particle> GetFoundParticles() const;
private:
	uint32_t m_uMaskRadius;
	uint32_t m_uFeatureRadius;
	uint32_t m_uMinStackCount;
	uint32_t m_uMaxStackCount;
	uint32_t m_uNeighborRadius;
	uint32_t m_nMaxLevel;

	dImg m_dCircleMask; // on the host for now
	dImg m_dRadXKernel;
	dImg m_dRadYKernel;
	dImg m_dRadSqKernel;

	size_t m_uCurPrevParticleCount;
	dParticleVec md_PrevParticleVec;
};