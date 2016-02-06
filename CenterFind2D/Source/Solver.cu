#include "CenterFind.h"

#include <set>

#include <cuda_runtime.h>

#include "ThrustOps.cuh"

// We need to change this so that most of the solving mechanisms are accessible via CUDA

Solver::Solver() :
m_uMaskRadius( 0 ),
m_uFeatureRadius( 0 ),
m_uMaxStackCount( 0 ),
m_uNeighborRadius( 0 )
{
}


struct get_int2 : public thrust::unary_function<unsigned int, int2 >
{
	int N;
	get_int2( int n ) :
		N( n )
	{
	}
	__host__ __device__
		inline int2 operator()( unsigned int idx )
	{
		int x = idx % N;
		int y = idx / N;;
		return make_int2( x, y );
	}
};

// Kernel for initializing solver kernels...kernel kernel
__global__
void createSolverKernels( int radius, float * circ, float * x, float * y, float * sq )
{
	// Pretty small sizes here
	int idx_X = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_Y = threadIdx.y + blockDim.y * blockIdx.y;
	int diameter = 2 * radius + 1;
	int idx = idx_X + idx_Y * diameter;

	x[idx] = idx_X + 1;
	y[idx] = idx_Y + 1;
	sq[idx] = powf( x[idx] - idx_X, 2 ) + powf( y[idx] - idx_Y, 2 );

}

Solver::Solver( uint32_t mR, uint32_t fR, uint32_t minSC, uint32_t maxSC, uint32_t nR ) :
m_uMaskRadius( mR ),
m_uFeatureRadius( fR ),
m_uMinStackCount( minSC ),
m_uMaxStackCount( maxSC ),
m_uNeighborRadius( nR )
{
	// Neighbor region diameter
	int diameter = 2 * m_uMaskRadius + 1;

	// Create GpuMats and initialize via kernel
	auto makeContinuousGmat = [diameter] () {
		GpuMat g = cv::cuda::createContinuous( cv::Size( diameter, diameter ), CV_32F );
		assert( g.isContinuous() && "We need contiguous arrays here" );
		return g;
	};

	m_CircleMask = makeContinuousGmat();
	m_RadXKernel = makeContinuousGmat();
	m_RadYKernel = makeContinuousGmat();
	m_RadSqKernel = makeContinuousGmat();

	// make data
	dim3 gridSize( 1 ), blockSize( diameter, diameter );
	createSolverKernels << < gridSize, blockSize >> >( (int) m_uMaskRadius, m_CircleMask.ptr<float>(), m_RadSqKernel.ptr<float>(), m_RadYKernel.ptr<float>(), m_RadSqKernel.ptr<float>() );

	// Make host mats
	cv::Mat h_Circ( cv::Size( diameter, diameter ), CV_32F, 0.f );
	cv::Mat h_RX = h_Circ;
	cv::Mat h_RY = h_Circ;
	cv::Mat h_R2 = h_Circ;

	// set up circle mask
	cv::circle( h_Circ, cv::Point( m_uMaskRadius, m_uMaskRadius ), m_uMaskRadius, 1.f, -1 );

	// set up Rx and part of r2
	for ( int i = 0; i < diameter; i++ )
	{
		for ( int j = 0; j < diameter; j++ )
		{
			h_RX.at<float>( i, j ) = float( j + 1 );
			h_R2.at<float>( i, j ) += float( pow( j - m_uMaskRadius, 2 ) );
		}
	}

	// set up Ry and the rest of r2
	for ( int i = 0; i < diameter; i++ )
	{
		for ( int j = 0; j < diameter; j++ )
		{
			h_RY.at<float>( i, j ) = float( i + 1 );
			h_R2.at<float>( i, j ) += float( pow( i - m_uMaskRadius, 2 ) );
		}
	}

	// I forget what these do...
	cv::threshold( h_R2, h_R2, pow( (double) m_uMaskRadius, 2 ), 1, cv::THRESH_TOZERO_INV );
	cv::multiply( h_RX, h_Circ, h_RX );
	cv::multiply( h_RY, h_Circ, h_RY );

	auto helper = [] ( cv::Mat& m ) {
		GpuMat g = cv::cuda::createContinuous( m.size(), m.type() );
		if ( g.isContinuous() == false )
		{
			// ruh roh
		}
		// copy memory
		return g;
	};

	// Create contiguous GPU Mats for these

	m_CircleMask = cv::cuda::createContinuous( h_Circ.size(), h_Circ.type() );

	// copy these to contiguous GpuMats
	m_CircleMask.upload( h_Circ );
	m_RadXKernel.upload( h_RX );
	m_RadYKernel.upload( h_RY );
	m_RadSqKernel.upload( h_R2 );

}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/tabulate.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

struct is_nonzero : public thrust::unary_function<thrust::tuple<unsigned char, int2>, bool >
{
	__host__ __device__
		inline bool operator()( const thrust::tuple<unsigned char, int2>& t )
	{
		return t.get<0>() != 0;
	}
};

__host__ __device__
int pLoc2Zcode( float2 pLoc, int max_level, int N )
{
	float xMin = 0, yMin = 0, xMax = N, yMax = N;

	int result = 0;

	for ( int level = 1; level <= max_level; level++ )
	{
		// Classify in x-direction
		float xmid = 0.5f * ( xMin + xMax );
		int x_hi_half = ( p.x < xmid ) ? 0 : 1;

		// Push the bit into the result as we build it
		result |= x_hi_half;
		result <<= 1;

		// Classify in y-direction
		float ymid = 0.5f * ( yMin + yMax );
		int y_hi_half = ( p.y < ymid ) ? 0 : 1;

		// Push the bit into the result as we build it
		result |= y_hi_half;
		result <<= 1;

		// Shrink the bounding box, still encapsulating the point
		xMin = ( x_hi_half ) ? xmid : xMin;
		xMax = ( x_hi_half ) ? xMax : xmid;
		yMin = ( y_hi_half ) ? ymid : yMin;
		yMax = ( y_hi_half ) ? yMax : ymid;
	}

	result >>= 1;
	return result;
}

struct Particle
{
	float2 pos;
	float intensity;
	int zCode;

	__host__ __device__
	Particle( float2 p, float i, int z ) :
		pos( p ),
		intensity( i ),
		zCode( z )
	{
	}
};

struct ParticleComp
{
	__host__ __device__
	bool operator()( Particle a, Particle b )
	{
		return a.zCode < b.zCode;
	}
};

// This is the second filter; once we know a particle is non-zero, we have to do a local sum around it
// to determine its "mass" (or intensity, not really sure)
struct GetParticle
{
	// Kernel Radius
	int kernelRadius;

	// Image dimensions
	int N;

	// offset kernels
	float * circKernel;
	float * xKernel;
	float * yKernel;
	float * sqKernel;

	// The actual reference images we multiply against
	float * lmImg;

	__host__ __device__
	GetParticle( int kD, int N, float * lmImg, float * cK, float * xK, float * yK, float * sqK ) :
		kernelRadius( kD ),
		N( N ),
		lmImg( lmImg ),
		circKernel( cK ),
		xKernel( xK ),
		yKernel( yK ),
		sqKernel( sqK )
	{
	}
	

	__host__ __device__
	Particle operator()( int idx )
	{
		// This would be the 2-d pixel location
		int2 loc2D = get_int2( N )( idx );

		// Center of the sum region
		float * center = &lmImg[idx];
		float total_mass( 0 );
		float x_Offset( 0 ), y_Offset( 0 ), sq_Offset( 0 );

		// I need to do the arithmetic that lets me loop through the square around center

		// Get the total mass and unnormalized x,y,sq offsets
		for ( int i = 0; i < 2 * kernelRadius + 1; i++ )
		{
			total_mass += circKernel[i] * center[i];
			x_Offset += xKernel[i] * center[i];
			y_Offset += yKernel[i] * center[i];
			sq_Offset += sqKernel[i] * center[i];
		}

		x_Offset /= total_mass;
		y_Offset /= total_mass;
		sq_Offset /= total_mass;

		// Compute x,y positions
		float xVal = x_Offset + loc2D.x;
		float yVal = x_Offset + loc2D.y;
		float r2_val = sq_Offset;

		// particle location and z code
		float2 pLoc = make_float2( xVal, yVal );
		int zCode = pLoc2Zcode( pLoc, 3, N );

		// Construct and return particle
		Particle p( pLoc, total_mass, zCode );
		return p;
	}
};

uint32_t Solver::FindParticles( Datum& D )
{
	// The particle image is contiguous, so let's find all particle locations and store their index in the image
	int N = D.d_ParticleImg.size().area();
	thrust::device_vector<int> d_ParticleIndices( N );

	// First make a device vector out of the existing particle image
	thrust::device_ptr<unsigned char> d_ParticleImgPtr( D.d_ParticleImg.ptr() );
	thrust::device_vector<unsigned char> d_ParticleImgVec( d_ParticleImgPtr, d_ParticleImgPtr + N );

	// Now we must zip the iterators such that every time we find a non-zero particle pixel, we're also given its location
	auto locFindItBegin = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.begin(), thrust::counting_iterator<int>(0) ) );
	auto locFindItEnd = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.end(), thrust::counting_iterator<int>( N ) ) );
	
	// The output iterator throws away the unsigned char img pixel values using a discard iterator, so we're left with the int2s
	auto locFindItOutput = thrust::make_zip_iterator( thrust::make_tuple( thrust::discard_iterator<>(), d_ParticleIndices.begin() ) );

	// Stream compact locations
	auto lastParticleIt = thrust::copy_if( locFindItBegin, locFindItEnd, locFindItOutput, is_nonzero() ); // is this legit? if not is_nonzero() works
	size_t numParticles = lastParticleIt - locFindItOutput;

	// Also, if you ever decide to display the particle locations on an image, here are the 2-d locations
	thrust::device_vector<int2> d_2DParticleLocations( numParticles );
	thrust::transform( d_ParticleIndices.begin(), d_ParticleIndices.end(), d_2DParticleLocations.begin(), get_int2( sqrt( N + 0.1 ) ) );

	// For each newly found particle, we can now transform it into a real particle
	// In order to do that we'll need some info about the lm img
	float * lmImg = D.d_LocalMaxImg.ptr<float>();
	float * circKern = m_CircleMask.ptr<float>();
	float * xKern = m_RadXKernel.ptr<float>();
	float * yKern = m_RadYKernel.ptr<float>();
	float * sqKern = m_RadSqKernel.ptr<float>();
	GetParticle gPOp( (int)m_uMaskRadius, (int)sqrt( N + 0.1 ), lmImg, circKern, xKern, yKern, sqKern );
	thrust::device_vector<Particle> d_ParticleVec( numParticles );
	thrust::transform( d_ParticleIndices.begin(), d_ParticleIndices.end(), d_ParticleVec.begin(), gPOp );

	// This is a dummy vector that would contain all previously found particles (a work in progress), sorted by their z-code
	thrust::device_vector<Particle> d_PreviouslyFoundParticleVec;

	// Find the range of previously found particles that could match our newly found particles
	thrust::device_vector<int> d_PrevParticleLB( numParticles ), d_PrevParticleUB( numParticles );
	thrust::lower_bound( d_PreviouslyFoundParticleVec.begin(), d_PreviouslyFoundParticleVec.end(), d_PreviouslyFoundParticleVec.begin(), d_PreviouslyFoundParticleVec.end(), ParticleComp() );
	thrust::upper_bound( d_PreviouslyFoundParticleVec.begin(), d_PreviouslyFoundParticleVec.end(), d_PreviouslyFoundParticleVec.begin(), d_PreviouslyFoundParticleVec.end(), ParticleComp() );
}