#include <cuda_runtime_api.h>

#include "Solver.cuh"

template <Particle::State PS>
struct IsParticleInState
{
	__host__ __device__
		bool operator()( const Particle p )
	{
		return p.pState == PS;
	}
};


// Turn an x,y coord into a grid index given the image dimension N
// and maximum subdivisions of that image m (N is a power of 2)
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

// Invoke the above given a particle
__host__ __device__
int pixelToGridIdx( Particle p, int N, int m )
{
	return pixelToGridIdx( p.x, p.y, N, m );
}

// Our particle to grid index operator
struct PixelToGridIdx : public thrust::unary_function < Particle, int >
{
	int N; // Image size
	int M; // division level

	PixelToGridIdx( int n, int m ) :N( n ), M( m ) {}

	__host__ __device__
		int operator()( Particle p )
	{
		// Just call the function
		return pixelToGridIdx( p, N, M );
	}
};

// Particle removal predicate
struct MaybeRemoveParticle
{
	int sliceIdx;	// Current slice index
	int minSlices;	// The minimum # of slices we require to contribute to a particle
	MaybeRemoveParticle( int s, int m ) : sliceIdx( s ), minSlices( m ) {}

	__host__ __device__
		bool operator()( const Particle p )
	{
		// If the particle is 2 (should be var) slices away from us, and it isn't in a sever state or is but has too few particles, return true
		return ( sliceIdx - p.lastContributingsliceIdx > 2 && ( p.pState != Particle::State::SEVER || p.nContributingParticles < minSlices ) );
	}
};

// Particle detection predicate
struct IsParticleAtIdx
{
	int N;			// Image dimension
	int featureRad;	// Image feature radius

	IsParticleAtIdx( int n, int k ) : N( n ), featureRad( k ) {}

	template <typename tuple_t>
	__host__ __device__
		bool operator()( tuple_t T )
	{
		// Unpack tuple
		unsigned char val = thrust::get<0>( T );
		int idx = thrust::get<1>( T );

		// get xy coords from pixel index
		int x = idx % N;
		int y = idx / N;

		// We care if the pixel is nonzero and its within the kernel radius
		return ( val != 0 ) && ( x > featureRad ) && ( y > featureRad ) && ( x + featureRad < N ) && ( y + featureRad < N );
	}
};

// Simple predicate that detects whether a pointer in this tuple is null
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

// Opposite of above
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

// Turn an index at which we've detected a particle (above) into a real particle
struct MakeParticleFromIdx
{
	int sliceIdx;		// Current slice index
	int kernelRad;		// Kernel (mask) radius
	int N;				// Image dimension

	float * lmImg;			// pointer to image in which particle was detected
	float * circKernel;		// pointer to circle mask kernel
	float * rxKernel;		// pointer to x offset kernel
	float * ryKernel;		// pointer to y offset kernel
	float * rSqKernel;		// pointer to r2 kernel

	MakeParticleFromIdx( int sIdx, int n, int kRad, float * lm, float * cK, float * xK, float * yK, float * sqK ) :
		sliceIdx( sIdx ),
		N( n ),
		kernelRad( kRad ),
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
		float x_val = float( x ) + x_offset / total_mass;
		float y_val = float( y ) + y_offset / total_mass;

		// Construct particle and return
		Particle p( x_val, y_val, total_mass, sliceIdx );
		return p;
	}
};

// This operator searches through the grid cells of our previous particle and tries to find a match
struct ParticleMatcher
{
	int N;					// Image dimension
	int M;					// Maximum # of subdivisions
	int sliceIdx;			// current slice index
	int maxStackCount;		// max # of slices that can contribute to a particle
	float neighborRadius;		// radius around which we search for matches

	int * cellLowerBound;		// Pointer to lower bound of prev particle range
	int * cellUpperBound;		// pointer to upper bound of prev particle range

	Particle* prevParticles;	// Pointer to the vector of previous particles

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

		// See if we need to search neighbors

		// But we always search at least one
		cellIndices[0] = pixelToGridIdx( newParticle, N, M );

		// Neighbors to follow
		const Particle * pBestMatch = nullptr;
		for ( int c = 0; cellIndices[c] >= 0; c++ )
		{
			// It would be nice to parallelize around this, but probably not worth it
			int cellIdx = cellIndices[c];
			int lower = cellLowerBound[cellIdx];
			int upper = cellUpperBound[cellIdx];
			for ( int p = lower; p < upper; p++ )
			{
				// Make reference to particle
				const Particle& oldParticle = prevParticles[p];

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
							pBestMatch = &prevParticles[p];
					}
					else
						pBestMatch = &prevParticles[p];
				}
			}
		}

		// Could check sever state here
		return (Particle *) pBestMatch;
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

// Used to sort particles by their grid index
// This could be phased out if I could get sort to work with a transform iterator
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