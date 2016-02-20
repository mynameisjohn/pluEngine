#include "Solver.cuh"
#include "SolverOperators.cuh"

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
m_uMinSliceCount( 0 ),
m_uMaxSliceCount( 0 ),
m_uNeighborRadius( 0 ),
m_uCurPrevParticleCount( 0 )
{
}

Solver::Solver( int mR, int fR, int minSC, int maxSC, int nR ) :
m_uMaskRadius( mR ),
m_uMaxLevel( 3 ),
m_uFeatureRadius( fR ),
m_uMinSliceCount( minSC ),
m_uMaxSliceCount( maxSC ),
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

	/// For host debugging
	//h_Circ.copyTo( m_dCircleMask );
	//h_RX.copyTo( m_dRadXKernel );
	//h_RY.copyTo( m_dRadYKernel );
	//h_R2.copyTo( m_dRadSqKernel );

	// Upload to continuous gpu mats
	m_dCircleMask = getContinuousGpuMat( h_Circ );
	m_dRadXKernel = getContinuousGpuMat( h_RX );
	m_dRadYKernel = getContinuousGpuMat( h_RY );
	m_dRadSqKernel = getContinuousGpuMat( h_R2 );
}

template <typename ... Args>
auto makeZipIt( const Args&... args ) -> decltype( thrust::make_zip_iterator( thrust::make_tuple( args... ) ) )
{
	return thrust::make_zip_iterator( thrust::make_tuple( args... ) );
}

// This function removes particles from the vector of previously found particles if they 
// pass the predicate MaybeRemoveParticle
size_t Solver::cullExistingParticles( int curSliceIdx )
{
	size_t u_preremovePrevParticleCount = m_uCurPrevParticleCount;
	auto itLastPrevParticleEnd = m_dPrevParticleVec.begin() + m_uCurPrevParticleCount;
	auto itCurPrevParticleEnd = thrust::remove_if( m_dPrevParticleVec.begin(), itLastPrevParticleEnd, MaybeRemoveParticle( curSliceIdx, m_uMinSliceCount ) );
	m_uCurPrevParticleCount = itCurPrevParticleEnd - m_dPrevParticleVec.begin();
	size_t nRemovedParticles = u_preremovePrevParticleCount - m_uCurPrevParticleCount;

	return nRemovedParticles;
}

// Given the processed particle image, this function finds the particle locations and returns a vector of Particle objects
Solver::ParticleVec Solver::findNewParticles( UcharVec& d_ParticleImgVec, Floatptr pThreshImg, int N, int sliceIdx )
{
	// Create pointers to our kernels
	Floatptr d_pCirleKernel( (float *) m_dCircleMask.data );
	Floatptr d_pRxKernel( (float *) m_dRadXKernel.data );
	Floatptr d_pRyKernel( (float *) m_dRadYKernel.data );
	Floatptr d_pR2Kernel( (float *) m_dRadSqKernel.data );

	// For each pixel in the particle image, we care if it's nonzero and if it's far enough from the edges
	// So we need its index (transformable into twoD pos) and its value, which we zip
	auto itDetectParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.begin(), thrust::counting_iterator<int>( 0 ) ) );
	auto itDetectParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.end(), thrust::counting_iterator<int>( N*N ) ) );

	// Then, if the particle fits our criteria, we copy its index (from the counting iterator) into this vector, and discard the uchar
	IntVec d_NewParticleIndicesVec( N*N );
	auto itFirstNewParticle = thrust::make_zip_iterator( thrust::make_tuple( thrust::discard_iterator<>(), d_NewParticleIndicesVec.begin() ) );
	auto itLastNewParticle = thrust::copy_if( itDetectParticleBegin, itDetectParticleEnd, itFirstNewParticle, IsParticleAtIdx( N, m_uFeatureRadius ) );
	size_t newParticleCount = itLastNewParticle - itFirstNewParticle;

	// Now transform each index into a particle by looking at values inside the lmimg and using the kernels
	ParticleVec d_NewParticleVec( newParticleCount );
	thrust::transform( d_NewParticleIndicesVec.begin(), d_NewParticleIndicesVec.begin() + newParticleCount, d_NewParticleVec.begin(),
					   MakeParticleFromIdx( sliceIdx, N, m_uMaskRadius, pThreshImg.get(), d_pCirleKernel.get(), d_pRxKernel.get(), d_pRyKernel.get(), d_pR2Kernel.get() ) );

	return d_NewParticleVec;
}

// This function recreates the grid cell ranges given the current container of previous particles
void Solver::createGridCells( int N )
{
	// We don't bother if there are no previous particles
	if ( m_dPrevParticleVec.empty() )
		return;

	// If our grid cell vectors are empty, create them now
	if ( m_dGridCellLowerBoundVec.empty() || m_dGridCellUpperBoundVec.empty() )
	{
		const int cellSize = N >> m_uMaxLevel;
		const int cellCount = N / cellSize;
		const int nTotalCells = cellCount * cellCount;
		m_dGridCellLowerBoundVec.resize( nTotalCells );
		m_dGridCellUpperBoundVec.resize( nTotalCells );
	}

	// Some typedefs, we use a transform iterator to convert particles into indices
	using particleIter = ParticleVec::iterator;
	using pixelToGridIdxIter = thrust::transform_iterator < PixelToGridIdx, particleIter >;

	// Create an iterator to the end of our current previous particle container (might not be m_dPrevParticleVec.end())
	auto itCurPrevParticleEnd = m_dPrevParticleVec.begin() + m_uCurPrevParticleCount;

	// Create the transform iterator that iterates over our previous particles and returns their grid indices
	pixelToGridIdxIter itPrevParticleBegin = thrust::make_transform_iterator<PixelToGridIdx, particleIter>( m_dPrevParticleVec.begin(), PixelToGridIdx( N, m_uMaxLevel ) );
	pixelToGridIdxIter itPrevParticleEnd = thrust::make_transform_iterator<PixelToGridIdx, particleIter>( itCurPrevParticleEnd, PixelToGridIdx( N, m_uMaxLevel ) );

	// Find the ranges of previous particless
	const size_t nTotalCells = m_dGridCellLowerBoundVec.size();
	thrust::lower_bound( itPrevParticleBegin, itPrevParticleEnd, thrust::counting_iterator<int>( 0 ), thrust::counting_iterator<int>( nTotalCells ), m_dGridCellLowerBoundVec.begin() );
	thrust::upper_bound( itPrevParticleBegin, itPrevParticleEnd, thrust::counting_iterator<int>( 0 ), thrust::counting_iterator<int>( nTotalCells ), m_dGridCellUpperBoundVec.begin() );
}

// For each new particle, given the range of previous particles to search through, find the best match and return a pointer to its address
// If the pointer is null, then no match was found
Solver::ParticlePtrVec Solver::findParticleMatches( ParticleVec& d_NewParticleVec, int N, int sliceIdx )
{
	ParticlePtrVec d_ParticleMatchVec( d_NewParticleVec.size(), (Particle *)nullptr );

	// Only go through this is there are cells we could match with
	if ( m_dPrevParticleVec.empty() == false )
		thrust::transform( d_NewParticleVec.begin(), d_NewParticleVec.end(), d_ParticleMatchVec.begin(),
		ParticleMatcher( N, m_uMaxLevel, sliceIdx, m_uMaxSliceCount, m_dGridCellLowerBoundVec.size(), m_uNeighborRadius, m_dGridCellLowerBoundVec.data().get(), m_dGridCellUpperBoundVec.data().get(), m_dPrevParticleVec.data().get() ) );

	return d_ParticleMatchVec;
}

// For every matched particle, update its intensity state / position
void Solver::updateMatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec, int sliceIdx )
{
	// Zip the pointer vec and newparticle vec
	auto itNewParticleToMatchedParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.begin(), d_ParticleMatchVec.begin() ) );
	auto itNewParticleToMatchedParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.end(), d_ParticleMatchVec.end() ) );

	// If there was a match, update the intensity state. I don't know how to do a for_each_if other than a transform_if that discards the output
	thrust::transform_if( itNewParticleToMatchedParticleBegin, itNewParticleToMatchedParticleEnd, thrust::discard_iterator<>(), UpdateMatchedParticle( sliceIdx ), CheckIfMatchIsNotNull() );

#if _DEBUG
	// Useful for me to know how these start to spread out on debug
	//size_t numInNoMatch = thrust::count_if( m_dPrevParticleVec.begin(), m_dPrevParticleVec.end(), IsParticleInState<Particle::State::NO_MATCH>() );
	//size_t numInIncreasing = thrust::count_if( m_dPrevParticleVec.begin(), m_dPrevParticleVec.end(), IsParticleInState<Particle::State::INCREASING>() );
	//size_t numInDecreasing = thrust::count_if( m_dPrevParticleVec.begin(), m_dPrevParticleVec.end(), IsParticleInState<Particle::State::DECREASING>() );
	//size_t numInSever = thrust::count_if( m_dPrevParticleVec.begin(), m_dPrevParticleVec.end(), IsParticleInState<Particle::State::SEVER>() );
#endif
}

// For the particles that weren't matched, stream compact them into a vector and return it
Solver::ParticleVec Solver::consolidateUnmatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec )
{
	// Zip the pointer vec and newparticle vec
	auto itNewParticleToMatchedParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.begin(), d_ParticleMatchVec.begin() ) );
	auto itNewParticleToMatchedParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.end(), d_ParticleMatchVec.end() ) );

	// Copy all unmatched particles into a new vector; we copy a tuple of new particles and pointers to matches, discarding the pointers
	ParticleVec d_UnmatchedParticleVec( d_NewParticleVec.size() );
	auto itNewParticleAndPrevParticleMatchBegin = thrust::make_zip_iterator( thrust::make_tuple( d_UnmatchedParticleVec.begin(), thrust::discard_iterator<>() ) );

	// Copy new particles if their corresponding match is null
	auto itNewParticleAndPrevParticleMatchEnd = thrust::copy_if( itNewParticleToMatchedParticleBegin, itNewParticleToMatchedParticleEnd, itNewParticleAndPrevParticleMatchBegin, CheckIfMatchIsNull() );
	size_t numUnmatchedParticles = itNewParticleAndPrevParticleMatchEnd - itNewParticleAndPrevParticleMatchBegin;

	// Size down and return
	d_UnmatchedParticleVec.resize( numUnmatchedParticles );
	return d_UnmatchedParticleVec;
}

// Given our previous particles and the newly found unmatched particles, merge them into a sorted container
void Solver::mergeUnmatchedParticles( ParticleVec& d_UnmatchedParticleVec, int N )
{
	// We have two options here; the easy option is to just tack these new particles onto the previous particle vector and sort the whole thing
	// alternatively you could set a flag in previous particles if the matching process caused them to move to a new grid cell and then treat those particles as unmatched
	// you could then sort the unmatched particles (relatively few compared to the count of previous particles) and then merge them into the prev particle vec, which is still sorted
	// Below is the first option, whihc was easier.

	// first make room for the new particles, if we need it
	size_t newPrevParticleCount = d_UnmatchedParticleVec.size() + m_uCurPrevParticleCount;
	if ( newPrevParticleCount > m_dPrevParticleVec.size() )
		m_dPrevParticleVec.resize( newPrevParticleCount );

	// copy unmatched particles onto the original end of the previous particle vec
	auto itNewParticleDest = m_dPrevParticleVec.begin() + m_uCurPrevParticleCount;
	auto itEndOfPrevParticles = thrust::copy( d_UnmatchedParticleVec.begin(), d_UnmatchedParticleVec.end(), itNewParticleDest );

	// Sort the whole thing
	thrust::sort( m_dPrevParticleVec.begin(), itEndOfPrevParticles, ParticleOrderingComp( N, m_uMaxLevel ) );
	m_uCurPrevParticleCount = newPrevParticleCount;
}

int Solver::FindParticles( Datum& D )
{
	// We assume the row and column dimensions are equal
	const int N = D.d_InputImg.rows;

	// Make a device vector out of the particle buffer pointer (it's contiguous)
	UcharPtr d_pParticleImgBufStart( (unsigned char *) D.d_ParticleImg.datastart );
	UcharPtr d_pParticleImgBufEnd( (unsigned char *) D.d_ParticleImg.dataend );
	UcharVec d_ParticleImgVec( d_pParticleImgBufStart, d_pParticleImgBufEnd );
	
	/// For host debugging
	//cv::Mat h_ThreshImg;
	//D.d_ThreshImg.download( h_ThreshImg );
	//Floatptr d_pLocalMaxImgBuf( h_ThreshImg.ptr<float>() );
	Floatptr d_pThreshImgBuf( (float *) D.d_ThreshImg.data );

	Floatptr d_pCirleKernel( (float *) m_dCircleMask.data );
	Floatptr d_pRxKernel( (float *) m_dRadXKernel.data );
	Floatptr d_pRyKernel( (float *) m_dRadYKernel.data );
	Floatptr d_pR2Kernel( (float *) m_dRadSqKernel.data );

	// Cull the herd
	size_t numParticlesRemoved = cullExistingParticles( D.sliceIdx );

	// Find new particles
	ParticleVec d_NewParticleVec = findNewParticles( d_ParticleImgVec, d_pThreshImgBuf, N, D.sliceIdx );
	size_t numParticlesFound = d_NewParticleVec.size();

	// Initialize grid cells given current container of previous particles
	createGridCells( N );

	// Tranform new particles into a vector of particle pointers; if they are null then no match was found (?)
	ParticlePtrVec d_ParticleMatchVec = findParticleMatches( d_NewParticleVec, N, D.sliceIdx );

	// For particles we were able to match, update their intensity states
	updateMatchedParticles( d_NewParticleVec, d_ParticleMatchVec, D.sliceIdx );

	// Copy all unmatched particles into a new vector; we copy a tuple of new particles and pointers to matches, discarding the pointers
	ParticleVec d_UnmatchedParticleVec = consolidateUnmatchedParticles( d_NewParticleVec, d_ParticleMatchVec );

	// Merge unmatched particles into our container, preserving grid index order
	mergeUnmatchedParticles( d_UnmatchedParticleVec, N );

	std::cout << "Slice Idx:\t" << D.sliceIdx << "\tNew Particles:\t" << numParticlesFound << "\tUnmatched Particles:\t" << d_UnmatchedParticleVec.size() << "\tFound Particles:\t" << m_uCurPrevParticleCount << "\tCulled Particles:\t" << numParticlesRemoved << std::endl;

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