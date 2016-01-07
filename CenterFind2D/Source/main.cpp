#include "CenterFind.h"

int main(int argc, char ** argv) {
	Engine E;
	E.Init("phi41pct_3D_6zoom_0001.tif", 1, 141);
	E.Execute();

	return 0;
}