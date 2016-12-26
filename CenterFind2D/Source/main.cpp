#include "CenterFind.h"

// Debug test exe, uses opencv GUI to set up parameters
int main(int argc, char ** argv) {
	// Construct engine
	Engine E;

	// Hard coded input (from testbed)
	std::list<std::string> liInput = { "phi41pct_3D_6zoom_0001.tif", "phi41pct_3D_6zoom_0002.tif", "phi41pct_3D_6zoom_0003.tif" };
	if (!E.Init(liInput, 1, 141))
			return -1;

	// Solve for particles
	int nParticlesFound = E.Execute();
	std::cout << nParticlesFound << " Particles found in input stack" << std::endl;

	return 0;
}
