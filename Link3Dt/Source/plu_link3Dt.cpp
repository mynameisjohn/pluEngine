//Peter J. Lu
//Copyright 2008 Peter J. Lu.
//http://www.peterlu.org
//plu@fas.harvard.edu
//
//This program is free software; you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation; either version 3 of the License, or (at
//your option) any later version.
//
//This program is distributed in the hope that it will be useful, but
//WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
//General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program; if not, see <http://www.gnu.org/licenses>.
//
//Additional permission under GNU GPL version 3 section 7
//
//If you modify this Program, or any covered work, by linking or
//combining it with MATLAB (or a modified version of that library),
//containing parts covered by the terms of MATLAB User License, the
//licensors of this Program grant you additional permission to convey
//the resulting work.

#include <ctime>
#include <string>
using namespace std;
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#define NUMCOLUMNS 6	//number of colums in INPUT data file
#define NUMROWS 1000000	//Number of rows for each stack (make large; increase if necessary)

string Date_Last_Updated="11 October 2009";
string Version_Number="4.5";

// Update History:
// July 17, 2005: MEL code removed; replaced by direct MEL script
// August 20, 2005: added normalized particle coordinates (0 to 1, instead of microns);
// fixed bug in row counting (make sure to subtract the last row from the counter;
// added comments and closed filestreams; put particle radius parameter in;
// added new error message, if dollar signs screw up data loading, say so.
// calculate proper volume fraction after accounting for boundaries
//
// August 27, 2005: Implemented multiple stacks
// August 31, 2005: Changed data input format to new 6-column list
// January 23, 2006: Added fourth exec_mode that forces single index to 1 on output, for use only with single stack [obsolete]
// 21 August 2006: Removed Legacy Maya Code; fixed crashing due to memory problems (properly cleans up memory in the while loop---should have fixed this before!)
// 02 August 2008: Changed execution modes to include Mass and Radius of Gyration information
// 23 August 2008 [4.3]: Kept maximum radius of gyration, updated file extensions to load data from "filestem_2DCenters.txt"
// 19 July 2009 [4.4]: Updated output to be more compatible with S(q) via exec_mode == 3
// 10 October 2009 [4.5]: Changed CLK_TCK to CLOCKS_PER_SEC for compatibility with Linux


/*
Particle Data Columns from input file:
0:  Stack number
1:  Frame (in stack) number
3:  Final X-position of particle, in pixels
4:  Final Y-position of particle, in pixels
7:  Mass (sum of brightness) of particle, in arbitrary units
8:  Radius of Gyration (sum of brightness * r^2) of particle, in arb. units
*/

int read_stack_from_file(ifstream &indatafile, float (*filedata)[NUMCOLUMNS], int &linesize) {
	//function reads the next 3D-stack's worth of particle position data;
	//returns number of particles; advances file pointer to beginning of next slice;
	//clears file data array, then fills with data from the file
	//Note that you have to pass the filestream by reference, otherwise it doesn't work!!
	int num_data_rows_in_stack = 0, i=1, j=0;
	int stacknumber = 0, tempstacknumber=0;
	//backup file pointer to beginning of current stack
	indatafile.seekg(-linesize,ios_base::cur);
	for(j=0; j<NUMCOLUMNS; j++) {
		indatafile >> filedata[0][j] ;
	}
	stacknumber = filedata[0][0];
	tempstacknumber = stacknumber;
	int pos_before_lineread = 0, pos_after_lineread = 0;;
	while(tempstacknumber==stacknumber && indatafile.eof()==false) {
		pos_before_lineread = indatafile.tellg();
		for(j=0; j<NUMCOLUMNS; j++) {
			indatafile >> filedata[i][j] ;
		}
		tempstacknumber = filedata[i][0];
		i++;
	}
	pos_after_lineread = indatafile.tellg();
	linesize = pos_after_lineread - pos_before_lineread;
	num_data_rows_in_stack = i-1;
	return num_data_rows_in_stack;
}

void print_row(float (*data)[10], const int row, ostream &out) {
	for(int i=0;i<NUMCOLUMNS;i++) {
		out << data[row][i] << "\t";
	}
	out << endl;
}

void print_input_datarow(float (*data)[NUMCOLUMNS], const int row, ostream &out) {
	for(int i=0;i<NUMCOLUMNS;i++) {
		out << data[row][i] << "\t";
	}
	out << endl;
}


int find_slice_rows(int (*slicerows)[3], float (*datalist)[10], const int numrows, const int numslices)
{
	//check for proper start of slice 1 at beginning of data set
	if((int) datalist[0][1] != 1) {
		cout << "Improper 2-D Data file: does not start with slice 1" << endl;
		cout << "If this is intended, please re-number slices, so that first slice is number 1" << endl;
		print_row(datalist,0,cout);
		return -1;
	}
	int sliceindex = 0;
	//initialize first row of first slice and last row of last slice
	slicerows[sliceindex][0] = 0;
	for(int i=1; i<numrows; i++) {
		if(datalist[i][1] > datalist[i-1][1]) {
			slicerows[sliceindex][1] = i-1;
			slicerows[sliceindex][2] = datalist[i-1][2];
			if(sliceindex < numslices) {
				slicerows[++sliceindex][0] = i;
			}
		}
	}
	slicerows[sliceindex][1] = numrows-1;
	//	for(int j=0; j<numslices; j++) {
	//		cout << j+1 << "\t" << slicerows[j][0] << "\t" << slicerows[j][1] << "\t" << slicerows[j][2] << endl;
	//	}
	return 0;
}

int main(int argc, char* argv[])
{
	clock_t starttime = clock();

	if (argc < 10) {
		cout << "Program to link 2-D centers into 3-D xyz data" << endl;
		cout << "(C)opyright 2009 Peter J. Lu and Hidekazu Oki" << endl;
		cout << endl << "****If you use this code, please cite in your publications: " << endl;
		cout << "P. J. Lu, P. A. Sims, H. Oki, J. B. Macarthur, and D. A. Weitz, " << endl;
		cout << "Target-locking acquisition with real-time confocal (TARC) microscopy," << endl;
		cout << "Optics Express Vol. 15, pp. 8702-8712 (2007)." << endl << endl;
		cout << "Version " << Version_Number << "; last updated " << Date_Last_Updated <<  endl << endl;
		cout << "plu_link3Dt.exe filestem max_r_dev min_slices max_slices boundary_reject xympp zmps radius exec_mode" << endl;
		cout << "2D Data file is named filestem_data.txt, and all parameters are the units (e.g. microns) used in that file." << endl;
		cout << "max_r_dev = maximum radial deviation in sucessive slices for same particle" << endl;
		cout << "min_slices = minimum number of slices in which a particle can appear" << endl;
		cout << "max_slices = maximum number of slices in which a particle can appear" << endl;
		cout << "boundary_reject = minimum distance away from all boundaries for a particle to be included" << endl;
		cout << "xympp = length in real units (e.g. microns) per pixel in x/y" << endl;
		cout << "zmps = interslice separation in z" << endl;
		cout << "radius = particle radius (estimate, for volume fraction calculation only)" << endl;
		cout << "exec_mode = 0: single text file; 1: +one file/stack with Rg and Mass information" << endl;
		cout << "		2: +one file/stack with Rg and Mass info; no check for z-intensity profile" << endl;
		cout << "		3: +one file/stack with position only; for, e.g., S(q) Fortran code" << endl;
		cout << endl << "Calculates positions for all particles in all stacks (i.e. multiple stacks)." << endl;
		return -1;
	}
	const float max_r_dev = atof(argv[2]);
	const float max_r2_dev = max_r_dev * max_r_dev;
	const int min_slices = atoi(argv[3]);
	const int max_slices = atoi(argv[4]);
	const float boundary_r = atof(argv[5]);
	const float xyfactor = atof(argv[6]);
	const float zfactor = atof(argv[7]);
	const float particle_radius = atof(argv[8]);
	const int exec_mode = atoi(argv[9]);

	//create filenames and filestreams
	string filestem = argv[1];
	//string infilename = filestem + "_data.txt";
	string infilename = filestem + "_2DCenters.txt";
	string outfilename = filestem + "_xyzt.txt";
	ifstream indatafile;		//input data file (output from 2D center-finding software)
	ofstream outalldatafile;		//output data file (z-averaged data)
	outalldatafile.open(outfilename.c_str(), ios::out);

	//log file: initial information
	string logfilename = filestem + "_xyzt_log.txt";
	ofstream logfile;		//output log file
	logfile.open(logfilename.c_str(), ios::out);
	logfile << "Program to link 2-D centers into 3-D xyzt data" << endl;
	logfile << "(C)opyright 2008 Peter J. Lu and Hidekazu Oki" << endl;
	logfile  << endl << "****If you use this code, please cite in your publications: " << endl;
	logfile  << "P. J. Lu, P. A. Sims, H. Oki, J. B. Macarthur, and D. A. Weitz, " << endl;
	logfile  << "Target-locking acquisition with real-time confocal (TARC) microscopy," << endl;
	logfile  << "Optics Express Vol. 15, pp. 8702-8712 (2007)." << endl << endl;
	logfile << "Version " << Version_Number << "; last updated " << Date_Last_Updated <<  endl << endl;
	logfile << "2D data input file: " << infilename << endl;
	logfile << "All-stack output data coordinate file: " << outfilename << endl;
	logfile << "This log file: " << logfilename << endl << endl;
	logfile << "Parameters for this analysis:" << endl;
	logfile  << "max_r_dev = maximum radial deviation in sucessive slices for same particle: " << max_r_dev << endl;
	logfile  << "min_slices = minimum number of slices in which a particle can appear: " << min_slices << endl;
	logfile  << "max_slices = maximum number of slices in which a particle can appear: " << max_slices << endl;
	logfile  << "boundary_reject = minimum distance away from all boundaries for a particle to be included: " << boundary_r << endl;
	logfile  << "xympp = length in real units (e.g. microns) per pixel in x/y: " << xyfactor << endl;
	logfile  << "zmps = interslice separation in z: " << zfactor << endl;
	logfile  << "radius = particle radius (estimate, for volume fraction calculation only): " << particle_radius << endl;
	logfile  << "exec_mode: ";
	if(exec_mode == 0) {
		logfile << "0--only single file";
	}
	if(exec_mode == 1) {
		logfile << "1--single file + one file per stack with gyration radius and mass information";
	}
	if(exec_mode == 2) {
		logfile << "2--single file, one file per stack with info; no checking of z-intensity profile";
	}
	if(exec_mode == 3) {
		logfile << "3--single file + one file per stack with only position information, plus extents in first line";
	}

	logfile << endl << endl;

	ofstream outStackdatafile;		//output data file (z-averaged data) for a single stack

	//Read in data
	//Want to create a list of xyz particle coordinates for each slice, and loop
	//Only want to read in enough of the file to process one stack at a time
	//so increment file counter only so that one slice is read into memory at a time
	float (*data_list)[NUMCOLUMNS] = new float[NUMROWS][NUMCOLUMNS];

	indatafile.open(infilename.c_str(),ios::in);
	if(!indatafile) {
		cerr << "Cannot open input data file." << endl;
		return -1;
	}

	//read first line to figure out how big each line is
	//need to back up one row, since while loop checks for differences in stack number
	float (*stackfirstrow)[NUMCOLUMNS] = new float[1][NUMCOLUMNS];
	for(int p=0; p<NUMCOLUMNS; p++) {
		indatafile >> stackfirstrow[0][p];
	}
	int linesize = indatafile.tellg();
	delete [] stackfirstrow;
	stackfirstrow = NULL;

	//main loop of program: loads one stack of data, does 3-D tracking, then repeats until done.
	while(indatafile.eof() == false) {

		float (*infiledata)[NUMCOLUMNS] = new float[NUMROWS][NUMCOLUMNS];
		int rows_in_stack =	read_stack_from_file(indatafile, infiledata, linesize);
		int stacknum = infiledata[0][0];

		ostringstream ss_stacknumber;
		ss_stacknumber.precision(3);
		ss_stacknumber << stacknum;
		string string_stacknumber = ss_stacknumber.str();
		if (exec_mode > 0) {
			string outStackfilename = filestem + "_" + string_stacknumber + "_xyz.txt";
			outStackdatafile.open(outStackfilename.c_str(), ios::out);
			logfile << "Stack output file: " << outStackfilename << endl;
		}

		logfile << "Stack " << stacknum << " has " << rows_in_stack << " rows." << endl;
		print_input_datarow(infiledata, 0, logfile);
		print_input_datarow(infiledata, rows_in_stack-1, logfile);

		//copy over relevant data only to new array
		float (*data_list)[10] = new float[rows_in_stack][10];
		int i=0;
		for(i=0; i<rows_in_stack; i++) {
			data_list[i][0]=infiledata[i][0];	//stack number
			data_list[i][1]=infiledata[i][1];	//slice number
			data_list[i][2]=0;					//obsolete
			data_list[i][3]=infiledata[i][2];	//x-coordinate
			data_list[i][4]=infiledata[i][3];	//y-coordinate
			data_list[i][7]=infiledata[i][4];	//integrated intensity (i.e. "mass" proxy)
			data_list[i][8]=infiledata[i][5];	//radius-of-gyration calculated from intensity
			data_list[i][5] = 0;	//replace column 5 (formerly x offset) with number of ABOVE slices (not counting present) that the particle occurs in
			data_list[i][6] = 0;	//replace column 6 (formerly y offset) with row number of particle in previous slice (0 if none)
			data_list[i][9] = 0;	//replace column 9 (formerly multiplicity) with number index of each particle (no need to keep mult. info)

		}

		//determine starting and ending row numbers for each slice
		const int num_slices = data_list[rows_in_stack-1][1];
		logfile << "Total slices: " << num_slices << endl;
		//new data structure has three columns:
		//[0] is start row number, [1] is end row number;
		//[2] is number of particles per slice; index is slice number +1
		int (*slice_rows)[3] = new int[num_slices+1][3];
		int status = find_slice_rows(slice_rows, data_list, rows_in_stack, num_slices);

		//initialize particle index for the particles in the topmost slice
		//remember: i is the particle index counter!!
		for(i=0; i<slice_rows[1][0]; i++) {
			data_list[i][9] = i;
		}

		//link up rows, the heart of the program
		float xpos=0, ypos=0, cur_xpos=0, cur_ypos=0, r2 = 0;
		int j=0, k=0, l=0;
		for(j=1; j<num_slices; j++) {
			//this loop is over slices
			for(k=slice_rows[j][0]; k<=slice_rows[j][1]; k++) {
				//read the current x and y positions of a given particle in the present slice
				cur_xpos = data_list[k][3];
				cur_ypos = data_list[k][4];
				for(l=slice_rows[j-1][0]; l<=slice_rows[j-1][1]; l++) {
					//loop through all particles in the previous slice, and see if any is within the radial threshold
					xpos = data_list[l][3];
					ypos = data_list[l][4];
					r2 = (cur_xpos - xpos) * (cur_xpos - xpos) + (cur_ypos - ypos) * (cur_ypos - ypos);
					if(r2 < max_r2_dev) {
						//check to see if current kth particle (in jth slice) is within the threshold of the lth particle in the (j-1)st slice
						//if so, assign the particle index, increment the slice count, and keep track of row number of lth particle
						data_list[k][9] = data_list[l][9];		//particle number index
						data_list[k][5] = data_list[l][5]+1;	//slice count
						data_list[k][6] = l;					//row number of lth particle
					}
				}
			}

			//if too many slices are considered the same particle (if the slice count is above max_slices), then
			//split into two particles, assign new indices, and continue

			//variables to hold intensity information for a given particle in a given slice, and for the same particle one slice up
			float current_intensity = 0, previous_intensity = 0;
			int previous_index = 0;

			for(k=slice_rows[j][0]; k<=slice_rows[j][1]; k++) {
				//check slices after minimum to see if intensity is going in the wrong direction (i.e. increasing)
				//Disable check for execution mode 2
				//this is NOT a hard cutoff (cf. hard cutoff below if things are getting ridiculous)
				if((int) data_list[k][5] >= min_slices && exec_mode != 2) {
					current_intensity = data_list[k][7];
					previous_index = data_list[k][6];
					previous_intensity = data_list[previous_index][7];

					//Check to see if intensity is INCREASING: if already through brightness maximum, should
					//monotonically decrease as function of z; an increase is a new particle.
					//In this case, assign new particle index and reset slice count.
					//Disable check for execution mode 2
					if(current_intensity > previous_intensity ) {
						data_list[k][9] = i;
						data_list[k][5] = 0;
						i++;
					}
				}

				//hard cutoff if too many slices still
				if((int) data_list[k][5] >= max_slices && exec_mode != 2) {
					data_list[k][9] = i;
					data_list[k][5] = 0;
					i++;
				}
				//if no match, assign new particle index (check to see if new particle; if so, assign new index)
				else if((int) data_list[k][9] == 0) {
					data_list[k][9] = i;
					i++;
				}
			}

		}

		//count total particles found in 3D
		int q=0, max_part_ind=0;
		for(q=0; q<rows_in_stack; q++) {
			if(data_list[q][9] > max_part_ind) {
				max_part_ind = data_list[q][9];
			}
		}
		logfile << "Total number of particle rows: " << max_part_ind << endl;

		//create new array data_zavg to hold averaged particle data
		float (*data_zavg)[6] = new float[max_part_ind+1][6];
		float weightfactor = 0;
		int current_row = 0, current_particle = 0;

		//initialize data_zavg to 0
		for (current_particle = 0; current_particle <= max_part_ind; current_particle++) {
			for(int r=0; r<6; r++) {
				data_zavg[current_particle][r] = 0;
			}
		}

		float radius_of_gyration = 0;
		//go through big row list, extract index, then calculate z-averaged position data in new list, as well as statistics
		for(current_row = 0; current_row < rows_in_stack; current_row++) {
			current_particle = data_list[current_row][9];
			weightfactor = data_list[current_row][7];	//using mass as the weighting factor
			data_zavg[current_particle][0] += weightfactor * data_list[current_row][3];
			data_zavg[current_particle][1] += weightfactor * data_list[current_row][4];
			data_zavg[current_particle][2] += weightfactor * data_list[current_row][1];
			data_zavg[current_particle][3] += weightfactor;	//total integrated intensity (i.e. mass)
			//keep maximum radius of gyration
			radius_of_gyration = xyfactor * data_list[current_row][8];
			if(data_zavg[current_particle][4] < radius_of_gyration) {
				data_zavg[current_particle][4] = radius_of_gyration;
			}
			data_zavg[current_particle][5] = data_list[current_row][5]+1;	//slice count
		}

		for (current_particle = 0; current_particle <= max_part_ind; current_particle++) {
			data_zavg[current_particle][0] = xyfactor * data_zavg[current_particle][0] / data_zavg[current_particle][3] ;
			data_zavg[current_particle][1] = xyfactor* data_zavg[current_particle][1] / data_zavg[current_particle][3] ;
			data_zavg[current_particle][2] = zfactor * data_zavg[current_particle][2] / data_zavg[current_particle][3] ;
		}

		//assemble final list of particles
		float (*data_zavg_filt)[6] = new float[max_part_ind+1][6];
		int filt_counter = 0;
		for(i=0; i<=max_part_ind; i++) {
			//conditions to include particle in list
			if((int) data_zavg[i][5] >= min_slices) {
				for(int j=0; j<6; j++) {
					data_zavg_filt[filt_counter][j] = data_zavg[i][j];
				}
				filt_counter++;
			}
		}
		const int num_particles = filt_counter;
		logfile << "Number of particles (after linkage, before boundary check): " << num_particles << endl;

		//calculate maximum and average values
		float minx = 1000, maxx = 0, miny = 1000, maxy = 0, minz = 1000, maxz = 0;
		for (j=0; j < num_particles; j++) {
			if(data_zavg_filt[j][0] < minx) {
				minx = data_zavg_filt[j][0];
			}
			if(data_zavg_filt[j][1] < miny) {
				miny = data_zavg_filt[j][1];
			}
			if(data_zavg_filt[j][2] < minz) {
				minz = data_zavg_filt[j][2];
			}
			if(data_zavg_filt[j][0] > maxx) {
				maxx = data_zavg_filt[j][0];
			}
			if(data_zavg_filt[j][1] > maxy) {
				maxy = data_zavg_filt[j][1];
			}
			if(data_zavg_filt[j][2] > maxz) {
				maxz = data_zavg_filt[j][2];
			}
		}

		float deltax = maxx - minx;
		float deltay = maxy - miny;
		float deltaz = maxz - minz;
		float xcenter = 0.5 * deltax;
		float ycenter = 0.5 * deltay;
		float zcenter = 0.5 * deltaz;
		int maxlength = 0;
		if(deltax > deltay && deltax > deltaz) {
			maxlength = deltax;
		}
		else if(deltay > deltaz && deltay > deltax) {
			maxlength = deltay;
		}

		else {
			maxlength = deltaz;
		}
		logfile << "Extents in x,y,z (before boundary removal):\t" << deltax << ", " << deltay << ", " << deltaz << endl;
		float single_particle_volume = 4 * 3.14159265 * particle_radius * particle_radius * particle_radius / 3;
		float system_volume = deltax * deltay * deltaz;
		logfile << "Volume fraction (before boundary removal): " << num_particles * single_particle_volume / system_volume << endl;

		maxlength -= 1.5*boundary_r;	//remove boundary distance from maximum length (should be twice the boundary_r, but leave a little wiggle room)
		int p, inside_boundary_counter = 0;
		float ib_minx = 1000, ib_maxx = 0, ib_miny = 1000, ib_maxy = 0, ib_minz = 1000, ib_maxz = 0;

		if(exec_mode == 3) {
			outStackdatafile << deltax << "\t";
			outStackdatafile << deltay << "\t";
			outStackdatafile << deltaz << "\t";
			outStackdatafile << endl;
		}

		for(p=0; p<num_particles; p++) {
			//check for boundary conditions: is particle too close to boundary? If not, don't print out
			//recalculate volume fraction for just particles inside the boundaries ('ib');
			if( (data_zavg_filt[p][0] > minx + boundary_r) && (data_zavg_filt[p][0] < maxx - boundary_r) &&
				(data_zavg_filt[p][1] > miny + boundary_r) && (data_zavg_filt[p][1] < maxy - boundary_r) &&
				(data_zavg_filt[p][2] > minz + boundary_r) && (data_zavg_filt[p][2] < maxz - boundary_r) ) {

				//keep track of how many particles
				inside_boundary_counter++;

				//output to _xyzt.txt data file (position and time info)
				outalldatafile << stacknum << "\t";
				for(i=0; i<3; i++) {
					outalldatafile << data_zavg_filt[p][i] << "\t";
				}
				outalldatafile << endl;

				if(exec_mode > 0) {
					//output to _xyz.txt data file (all info)
					//outStackdatafile << stacknum << "\t";
					for(i=0; i<3; i++) {
						outStackdatafile << data_zavg_filt[p][i] << "\t";
					}
					if (exec_mode < 3) {
						outStackdatafile << data_zavg_filt[p][4] << "\t";
						outStackdatafile << data_zavg_filt[p][3] << "\t";
						outStackdatafile << data_zavg_filt[p][5] << "\t";
					}
					outStackdatafile << endl;
				}

				//determine new extents, with only particles that are not within the boundaries of the edges of the imaged volume
				if(data_zavg_filt[p][0] < ib_minx) {
					ib_minx = data_zavg_filt[p][0];
				}
				if(data_zavg_filt[p][1] < ib_miny) {
					ib_miny = data_zavg_filt[p][1];
				}
				if(data_zavg_filt[p][2] < ib_minz) {
					ib_minz = data_zavg_filt[p][2];
				}
				if(data_zavg_filt[p][0] > ib_maxx) {
					ib_maxx = data_zavg_filt[p][0];
				}
				if(data_zavg_filt[p][1] > ib_maxy) {
					ib_maxy = data_zavg_filt[p][1];
				}
				if(data_zavg_filt[p][2] > ib_maxz) {
					ib_maxz = data_zavg_filt[p][2];
				}
			}
		}

		//determine size of region (excluding the boundaries) containing particles
		float ib_deltax = ib_maxx - ib_minx;
		float ib_deltay = ib_maxy - ib_miny;
		float ib_deltaz = ib_maxz - ib_minz;
		logfile << "minimum x: " << ib_minx << "\tminimum y: " << ib_miny << "\tminimum z: " << ib_minz << endl;
		logfile << "maximum x: " << ib_maxx << "\tmaximum y: " << ib_maxy << "\tmaximum z: " << ib_maxz << endl;

		logfile << "Extents in x,y,z (after boundary removal):\t" << ib_deltax << ", " << ib_deltay << ", " << ib_deltaz << endl;
		const int ib_num_particles = inside_boundary_counter;
		logfile << "Final number of particles (after boundary removal): " << ib_num_particles << endl;
		float ib_system_volume = ib_deltax * ib_deltay * ib_deltaz;
		logfile << "Volume fraction (after boundary removal): " << ib_num_particles * single_particle_volume / ib_system_volume << endl;

		//Housekeeping: clean up memory and close files
		delete [] data_list;
		data_list = NULL;
		delete [] infiledata;
		infiledata=NULL;
		delete [] slice_rows;
		slice_rows = NULL;
		delete [] data_zavg;
		data_zavg = NULL;
		delete [] data_zavg_filt;
		data_zavg_filt = NULL;

		if(exec_mode > 0) {
			outStackdatafile.close();
		}

		if(stacknum % 20 == 0) {
		cout << "Elapsed time after linking stack " << stacknum << ": " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl;
		}
		logfile << "Elapsed time after linking stack " << stacknum << ": " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl << endl;
	}

	//close data files
	indatafile.close();
	outalldatafile.close();

	delete [] data_list;
	data_list = NULL;

	//Show total time to run program
	cout << "Total time: " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl;
	logfile << "Total time: " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl;
	logfile.close();
	return 0;
}
