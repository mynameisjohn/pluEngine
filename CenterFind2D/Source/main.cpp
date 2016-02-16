#include "CenterFind.h"

int main(int argc, char ** argv) {
	Engine E;
	E.Init({ "phi41pct_3D_6zoom_0001.tif"/*, "phi41pct_3D_6zoom_0002.tif", "phi41pct_3D_6zoom_0003.tif"*/ }, 1, 2);
	E.Execute();

	return 0;
}