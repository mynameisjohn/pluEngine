#include "Solver.cuh"

#include <iterator>
#include <algorithm>
#include <set>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform.h>

#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

// x, y, i default to -1
__host__ __device__
Particle::Particle( float x, float y, float i, int idx ) :
z( idx ),
peakIntensity( i ),
nContributingParticles( 1 ),
lastContributingsliceIdx( idx ),
pState( Particle::State::NO_MATCH )
{
	this->x = x;
	this->y = y;
	this->i = i;
}

Solver::Solver() :
m_uMaskRadius( 0 ),
m_uFeatureRadius( 0 ),
m_uMaxStackCount( 0 ),
m_uNeighborRadius( 0 ),
m_uCurPrevParticleCount( 0 )
{
}

Solver::Solver( uint32_t mR, uint32_t fR, uint32_t minSC, uint32_t maxSC, uint32_t nR ) :
m_uMaskRadius( mR ),
m_nMaxLevel( 3 ),
m_uFeatureRadius( fR ),
m_uMinStackCount( minSC ),
m_uMaxStackCount( maxSC ),
m_uNeighborRadius( nR ),
m_uCurPrevParticleCount( 0 )
{
	// Neighbor region diameter
	int diameter = 2 * m_uMaskRadius + 1;

	// Make host mats
	cv::Mat h_Circ( cv::Size( diameter, diameter ), CV_32F, 0.f );
	cv::Mat h_RX( cv::Size( diameter, diameter ), CV_32F, 0.f );
	cv::Mat h_RY( cv::Size( diameter, diameter ), CV_32F, 0.f );
	cv::Mat h_R2( cv::Size( diameter, diameter ), CV_32F, 0.f );

	// set up circle mask
	cv::circle( h_Circ, cv::Point( m_uMaskRadius, m_uMaskRadius ), m_uMaskRadius, 1.f, -1 );

	// set up Rx and part of r2
	for ( int y = 0; y < diameter; y++ )
	{
		for ( int x = 0; x < diameter; x++ )
		{
			cv::Point p( x, y );
			h_RX.at<float>( p ) = x + 1;
			h_RY.at<float>( p ) = y + 1;
			h_R2.at<float>( p ) = pow( -(float) m_uMaskRadius + x, 2 ) + pow( -(float) m_uMaskRadius + y, 2 );
		}
	}

	// I forget what these do...
	cv::threshold( h_R2, h_R2, pow( (double) m_uMaskRadius, 2 ), 1, cv::THRESH_TOZERO_INV );
	cv::multiply( h_RX, h_Circ, h_RX );
	cv::multiply( h_RY, h_Circ, h_RY );

	// Upload to gpu mats
	//h_Circ.copyTo( m_dCircleMask );
	//h_RX.copyTo( m_dRadXKernel );
	//h_RY.copyTo( m_dRadYKernel );
	//h_R2.copyTo( m_dRadSqKernel );

	m_dCircleMask.upload( h_Circ );
	m_dRadXKernel.upload( h_RX );
	m_dRadYKernel.upload( h_RY );
	m_dRadSqKernel.upload( h_R2 );
}

__host__ __device__
int pixelToGridIdx( float x, float y, int N, int m )
{
	const int cellSize = N >> m;
	const int cellCount = N / cellSize;

	int cellX = x / cellSize;
	int cellY = y / cellSize;

	int cellIdx = cellX + cellCount * cellY;
	return cellIdx;
}

__host__ __device__
int pixelToGridIdx( Particle p, int N, int m )
{
	return pixelToGridIdx( p.x, p.y, N, m );
}

struct PixelToGridIdx : public thrust::unary_function < Particle, int >
{
	int N; // Image size
	int M; // division level

	PixelToGridIdx( int n, int m ) :N( n ), M( m ) {}

	__host__ __device__
	int operator()( const Particle& p )
	{
		return pixelToGridIdx( p.x, p.y, N, M );
	}
};

struct IsParticleAtIdx
{
	int N;
	int featureRad;
	IsParticleAtIdx( int n, int k ) : N( n ), featureRad( k ) {}

	template <typename tuple_t>
	__host__ __device__
	bool operator()( tuple_t T )
	{
		unsigned char val = thrust::get<0>( T );
		int idx = thrust::get<1>( T );
		int x = idx % N;
		int y = idx / N;
		
		// We care if the pixel is nonzero and its within the kernel radius
		return ( val != 0 ) && ( x > featureRad ) && ( y > featureRad ) && ( x + featureRad < N ) && ( y + featureRad < N );
	}
};

struct MakeParticleFromIdx
{
	int sliceIdx;
	int kernelRad;
	int N;

	float * lmImg;
	float * circKernel;
	float * rxKernel;
	float * ryKernel;
	float * rSqKernel;

	MakeParticleFromIdx( int sIdx, int n, int kRad, float * lm, float * cK, float * xK, float * yK, float * sqK ) :
		sliceIdx( sIdx ),
		N(n),
		kernelRad(kRad),
		lmImg( lm ),
		circKernel( cK ),
		rxKernel( xK ),
		ryKernel( yK ),
		rSqKernel( sqK )
	{
	}

	__host__ __device__
	Particle operator()( int idx )
	{
		// Grab x, y values
		int x = idx % N;
		int y = idx / N;

		// Make tmp pointers to our kernels and advance them as we iterate
		float * tmpCircKernPtr = circKernel;
		float * tmpXKernPtr = rxKernel;
		float * tmpYKernPtr = ryKernel;

		// To be calculated
		float total_mass( 0 );
		float x_offset( 0 ), y_offset( 0 );

		// Perform the multiplcations
		for ( int iY = -kernelRad; iY <= kernelRad; iY++ )
		{
			// For y, go down then up
			float * ptrY = &lmImg[idx + ( N * iY )];
			for ( int iX = -kernelRad; iX <= kernelRad; iX++ )
			{
				// Get the local max img value
				float lmImgVal = ptrY[iX]; 

				// Multiply by kernel, sum, advance kernel pointer
				total_mass += lmImgVal * ( *tmpCircKernPtr++ );
				x_offset += lmImgVal * ( *tmpXKernPtr++ );
				y_offset += lmImgVal * ( *tmpYKernPtr++ );
			}
		}

		// Calculate x val, y val
		float x_val = float(x) + x_offset / total_mass;
		float y_val = float(y) + y_offset / total_mass;

		// Construct particle and return
		Particle p( x_val, y_val, total_mass, sliceIdx );
		return p;
	}
};

struct ParticleMatcher
{
	int N;
	int M;
	int sliceIdx;
	int maxStackCount;
	float neighborRadius;

	int * cellLowerBound;
	int * cellUpperBound;

	Particle* prevParticles;

	ParticleMatcher( int n, int m, int s, int mSC, int nR, int * cLB, int * cUB, Particle * pP ) :
		N( n ),
		M( m ),
		sliceIdx( s ),
		maxStackCount( mSC ),
		neighborRadius( nR ),
		cellLowerBound( cLB ),
		cellUpperBound( cUB ),
		prevParticles( pP )
	{
	}

	// Returns null if no match is found
	__host__ __device__
	Particle * operator()( Particle newParticle )
	{
		// There are a total of 9 cells we might have to search. last is sentinel
		int cellIndices[10] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

		// But we always search at least one
		cellIndices[0] = pixelToGridIdx( newParticle, N, M );

		// Neighbors to follow
		Particle * pBestMatch = nullptr;
		for ( int c = 0; cellIndices[c] >= 0; c++ )
		{
			// It would be nice to parallelize around this, but probably not worth it
			int cellIdx = cellIndices[c];
			int lower = cellLowerBound[cellIdx];
			int upper = cellUpperBound[cellIdx];
			for ( int p = lower; p < upper; p++ )
			{
				Particle& oldParticle = prevParticles[p];

				// tooFar might not be necessary if I cull beforehand
				bool tooFar = ( sliceIdx - oldParticle.lastContributingsliceIdx != 1 );
				bool tooMany = ( oldParticle.nContributingParticles > maxStackCount );
				bool alreadyDone = ( oldParticle.pState == Particle::State::SEVER );
				if ( tooFar || tooMany || alreadyDone )
					continue;

				// See if the particle is within our range
				float dX = oldParticle.x - newParticle.x;
				float dY = oldParticle.y - newParticle.y;
				float distSq = pow( dX, 2 ) + pow( dY, 2 );

				if ( distSq < neighborRadius * neighborRadius )
				{
					// If there already was a match, see if this one is better
					if ( pBestMatch )
					{
						// Find the old distance
						dX = pBestMatch->x - newParticle.x;
						dY = pBestMatch->y - newParticle.y;

						// If this one is closer, assign it as the match
						if ( pow( dX, 2 ) + pow( dY, 2 ) > distSq )
							pBestMatch = &oldParticle;
					}
					else 
						pBestMatch = &oldParticle;
				}
			}
		}

		// Could check sever state here

		return pBestMatch;
	}
};

struct CheckIfMatchIsNotNull
{
	template <typename tuple_t>
	__host__ __device__
	bool operator()( const tuple_t T )
	{
		Particle * pMatch = thrust::get<1>( T );
		return pMatch != nullptr;
	}
};
struct CheckIfMatchIsNull
{
	template <typename tuple_t>
	__host__ __device__
		bool operator()( const tuple_t T )
	{
		Particle * pMatch = thrust::get<1>( T );
		return pMatch == nullptr;
	}
};

// This gets called on matched particles and handles intensity state logic
// You should ensure this is thread safe beforehand, somehow (remove duplicates? not really sure)
struct UpdateMatchedParticle
{
	int sliceIdx;

	UpdateMatchedParticle( int s ) : sliceIdx( s ) {}

	// This kind of thing could be parallelized in a smarter way, probably
	template <typename tuple_t>
	__host__ __device__
	int operator()( const tuple_t T )
	{
		Particle newParticle = thrust::get<0>( T );
		Particle * oldParticle = thrust::get<1>( T );
		switch ( oldParticle->pState )
		{
			case Particle::State::NO_MATCH:
				// Shouldn't ever get no match, but assign the state and fall through
				oldParticle->pState = Particle::State::INCREASING;
			case Particle::State::INCREASING:
				// If we're increasing, see if the new guy prompts a decrease
				// Should we check to see if more than one particle has contributed?
				if ( oldParticle->i > newParticle.i )
					oldParticle->pState = Particle::State::DECREASING;
				// Otherwise see if we should update the peak intensity and z position
				else if ( newParticle.i > oldParticle->peakIntensity )
				{
					oldParticle->peakIntensity = newParticle.i;
					oldParticle->z = (float) sliceIdx;
				}
				break;
			case Particle::State::DECREASING:
				// In this case, if it's still decreasing then fall through
				if ( oldParticle->i > newParticle.i )
					break;
				// If we're severing, assing the state and fall through
				oldParticle->pState = Particle::State::SEVER;

				// I could probably catch this earlier
			case Particle::State::SEVER:
				// Continue here (could catch this earlier)
				oldParticle = nullptr;
		}

		// could do this in yet another call, if you were so inclined
		// If we didn't sever and null out above
		if ( oldParticle != nullptr )
		{
			// It's a match, bump the particle count and compute an averaged position (?)
			oldParticle->nContributingParticles++;
			oldParticle->lastContributingsliceIdx = sliceIdx;

			// I don't know about the averaged position thing
			oldParticle->x = 0.5f * ( oldParticle->x + newParticle.x );
			oldParticle->y = 0.5f * ( oldParticle->y + newParticle.y );
		}

		return 0;
	}
};

struct IsParticleUnmatched
{
	__host__ __device__
	bool operator()( const Particle p )
	{
		return p.pState == Particle::State::NO_MATCH;
	}
};

struct ParticleOrderingComp
{
	int N, M;
	ParticleOrderingComp( int n, int m ) : N( n ), M( m ) {}

	__host__ __device__
	bool operator()( const Particle a, const Particle b )
	{
		return pixelToGridIdx( a, N, M ) < pixelToGridIdx( b, N, M );
	}
};

struct MaybeRemoveParticle
{
	int sliceIdx;
	int minSlices;
	MaybeRemoveParticle( int s, int m ) : sliceIdx( s ), minSlices( m ) {}

	__host__ __device__
	bool operator()(const Particle p)
	{
		return ( sliceIdx - p.lastContributingsliceIdx > 2 && ( p.pState != Particle::State::SEVER || p.nContributingParticles < minSlices ) );
	}
};

struct IsNonzero
{
	bool operator()( const unsigned char val )
	{
		return val != 0;
	}
};

uint32_t Solver::FindParticles( Datum& D )
{
	const int N = D.d_LocalMaxImg.rows;
	const int m = m_nMaxLevel;
	const int cellSize = N >> m;
	const int cellCount = N / cellSize;
	const int nTotalCells = cellCount * cellCount;

	dFloatptr d_pLocalMaxImgBuf( D.d_LocalMaxImg.ptr<float>() );
	dFloatptr d_pCirleKernel( m_dCircleMask.ptr<float>() );
	dFloatptr d_pRxKernel( m_dRadXKernel.ptr<float>() );
	dFloatptr d_pRyKernel( m_dRadYKernel.ptr<float>() );
	dFloatptr d_pR2Kernel( m_dRadSqKernel.ptr<float>() );

	// Cull the herd
	int minSlices = 3;
	auto itLastPrevParticleEnd = md_PrevParticleVec.begin() + m_uCurPrevParticleCount;
	itLastPrevParticleEnd = thrust::remove_if( md_PrevParticleVec.begin(), itLastPrevParticleEnd, MaybeRemoveParticle( D.sliceIdx, minSlices ) );
	m_uCurPrevParticleCount = itLastPrevParticleEnd - md_PrevParticleVec.begin();

	// Make a device vector out of the particle buffer pointer (it's contiguous)
	dUcharPtr d_pParticleImgBuf( D.d_ParticleImg.ptr<unsigned char>() );
	dUcharVec d_ParticleImgVec( d_pParticleImgBuf, d_pParticleImgBuf + D.d_ParticleImg.size().area() );

	// For each pixel in the particle image, we care if it's nonzero and if it's far enough from the edges
	// So we need its index (transformable into twoD pos) and its value
	auto itDetectParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.begin(), thrust::counting_iterator<int>( 0 ) ) );
	auto itDetectParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.end(), thrust::counting_iterator<int>( N*N ) ) );

	// Then, if the particle fits our criteria, we copy its index (from the counting iterator) into this vector, and discard the uchar
	dIntVec d_NewParticleIndicesVec( N*N );
	auto itFirstNewParticle = thrust::make_zip_iterator( thrust::make_tuple( thrust::discard_iterator<>(), d_NewParticleIndicesVec.begin() ) );
	auto itLastNewParticle = thrust::copy_if( itDetectParticleBegin, itDetectParticleEnd, itFirstNewParticle, IsParticleAtIdx( N, m_uFeatureRadius ) );
	size_t newParticleCount = itLastNewParticle - itFirstNewParticle;

	// Now transform each index into a particle by looking at values inside the lmimg and using the kernels
	dParticleVec d_NewParticleVec( newParticleCount );
	thrust::transform( d_NewParticleIndicesVec.begin(), d_NewParticleIndicesVec.begin() + newParticleCount, d_NewParticleVec.begin(),
					   MakeParticleFromIdx( D.sliceIdx, N, m_uMaskRadius, d_pLocalMaxImgBuf.get(), d_pCirleKernel.get(), d_pRxKernel.get(), d_pRyKernel.get(), d_pR2Kernel.get() ) );

	// The grid cell vec might be split into two vecs like this (they should also be class members, but I'll do that later)
	dIntVec d_GridCellLowerBoundsVec( nTotalCells ), d_GridCellUpperBoundsVec( nTotalCells );

	// Initialize grid cells
	using particleIter = dParticleVec::iterator;
	using pixelToGridIdxIter = thrust::transform_iterator < PixelToGridIdx, particleIter >;

	pixelToGridIdxIter itPrevParticleBegin = thrust::make_transform_iterator<PixelToGridIdx, particleIter>( md_PrevParticleVec.begin(), PixelToGridIdx( N, m ) );
	pixelToGridIdxIter itPrevParticleEnd = thrust::make_transform_iterator<PixelToGridIdx, particleIter>( itLastPrevParticleEnd, PixelToGridIdx( N, m ) );

	thrust::lower_bound( itPrevParticleBegin, itPrevParticleEnd, thrust::counting_iterator<int>( 0 ), thrust::counting_iterator<int>( nTotalCells ), d_GridCellLowerBoundsVec.begin() );
	thrust::upper_bound( itPrevParticleBegin, itPrevParticleEnd, thrust::counting_iterator<int>( 0 ), thrust::counting_iterator<int>( nTotalCells ), d_GridCellUpperBoundsVec.begin() );

	// Tranform new particles into a vector of particle pointers; if they are null then no match was found (?)
	dParticlePtrVec d_ParticleMatchVec( newParticleCount );
	thrust::transform( d_NewParticleVec.begin(), d_NewParticleVec.end(), d_ParticleMatchVec.begin(),
					   ParticleMatcher( N, m, D.sliceIdx, m_uMaxStackCount, m_uNeighborRadius, d_GridCellLowerBoundsVec.data().get(), d_GridCellUpperBoundsVec.data().get(), md_PrevParticleVec.data().get() ) );

	// Zip the pointer vec and newparticle vec
	auto itNewParticleToMatchedParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.begin(), d_ParticleMatchVec.begin() ) );
	auto itNewParticleToMatchedParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.end(), d_ParticleMatchVec.end() ) );

	// If there was a match, update the intensity state. I don't know how to do a for_each_if other than a transform_if that discards the output
	thrust::transform_if( itNewParticleToMatchedParticleBegin, itNewParticleToMatchedParticleEnd, thrust::discard_iterator<>(), UpdateMatchedParticle( D.sliceIdx ), CheckIfMatchIsNotNull() );

	// Copy all unmatched particles into a new vector; we copy a tuple of new particles and pointers to matches, discarding the pointers
	dParticleVec d_UnmatchedParticleVec( newParticleCount );
	auto itNewParticleAndPrevParticleMatchBegin = thrust::make_zip_iterator( thrust::make_tuple( d_UnmatchedParticleVec.begin(), thrust::discard_iterator<>() ) );

	// Copy new particles if their corresponding match is null
	auto itNewParticleAndPrevParticleMatchEnd = thrust::copy_if( itNewParticleToMatchedParticleBegin, itNewParticleToMatchedParticleEnd, itNewParticleAndPrevParticleMatchBegin, CheckIfMatchIsNull() );
	size_t numUnmatchedParticles = itNewParticleAndPrevParticleMatchEnd - itNewParticleAndPrevParticleMatchBegin;

	// We have two options here; the easy option is to just tack these new particles onto the previous particle vector and sort the whole thing
	// alternatively you could set a flag in previous particles if the matching process caused them to move to a new grid cell and then treat those particles as unmatched
	// you could then sort the unmatched particles (relatively few compared to the count of previous particles) and then merge them into the prev particle vec, which is still sorted

	// Option A
	{
		// first make room for the new particles, if we need it
		size_t newPrevParticleCount = numUnmatchedParticles + m_uCurPrevParticleCount;
		if ( newPrevParticleCount > md_PrevParticleVec.size() )
			md_PrevParticleVec.resize( newPrevParticleCount );

		// copy unmatched particles onto the original end of the previous particle vec
		auto itNewParticleDest = md_PrevParticleVec.begin() + m_uCurPrevParticleCount;
		thrust::copy( d_UnmatchedParticleVec.begin(), d_UnmatchedParticleVec.end(), itNewParticleDest );

		// Sort the whole thing
		thrust::sort( md_PrevParticleVec.begin(), md_PrevParticleVec.begin() + newPrevParticleCount, ParticleOrderingComp( N, m ) );
		m_uCurPrevParticleCount = newPrevParticleCount;
	}

	// Option B; NYI

	std::cout << "Slice Idx:\t" << D.sliceIdx << "\tNew Particles:\t" << newParticleCount << "\tUnmatched Particles:\t" << numUnmatchedParticles << "\tFound Particles:\t" << m_uCurPrevParticleCount << std::endl;

	return m_uCurPrevParticleCount;
}

std::vector<Particle> Solver::GetFoundParticles() const
{

	//int nParticles = std::count_if( m_vPrevParticles.begin(), m_vPrevParticles.end(), [] ( const Particle& p ) {
	//	return p.pState == Particle::State::SEVER && p.nContributingParticles > 2;
	//} );
	//std::cout << "Final particle count: " << nParticles << std::endl;

	std::vector<Particle> ret;
	//( m_vFoundParticles.size() );
	//std::transform( m_vFoundParticles.begin(), m_vFoundParticles.end(), ret.begin(),
	//				[] ( const ParticleStack& pS ) {return pS.GetRefinedParticle(); } );
	return ret;
}