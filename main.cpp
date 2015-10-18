#include "CenterFindEngine.h"

int main(int argc, char ** argv) {
	// Convert program args to CenterFind arguments
	std::array<std::string, 12> args;

	// Better be the right amount
	if (argc != args.size() + 1)
		return -1;
    
    

	// Make them strings
	for (int i=0; i < args.size(); i++)
		args[i] = std::string(argv[i+1]);
    
    for (int i=0; i<argc-1; i++)
        std::cout << i << ", " << args[i] << std::endl;

	// Construct parameters and engine
	CenterFindEngine::Parameters params(args);
	CenterFindEngine cF(params);

	// Execute engine, get results
	std::deque<CenterFindEngine::PMetricsVec> results = cF.Execute();

	// Send this to the input of link3Dt (or whatever comes next)

	return 0;
}
