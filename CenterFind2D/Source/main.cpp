#include "CenterFind.h"

int main(int argc, char ** argv) {
	Engine E;
	E.Init("test.tif", 1, 141);
	E.Execute();

	return 0;
}