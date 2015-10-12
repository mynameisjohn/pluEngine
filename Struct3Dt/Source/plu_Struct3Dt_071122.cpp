#include <time.h>
#include <string>
using namespace std;
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "plu_analysis_3Dt_071122.h"

#define NUMROWS 10000000	//Number of rows for each stack (make large; increase if necessary)
string Date_Last_Updated="22 November 2007";
string Version_Number="3.5";
#define EPSILON 0.00001

// Program combines various 3D analysis routines, and applies it to multiple stacks through time
// Update History:
// August 28, 2005: began version 1.0
// October 21, 2005: version 1.1: only percolated particles are used to calculate fractal dimension
// January 23, 2006: version 1.2: output statistics of largest cluster
// June 5, 2006: version 3.0: added number of chains (reconciled same version number to that of analysis code)
// 20 August 2006: version 3.1: Fixed bug so that 9th column of particle data now properly shows Rg (not Rg^2, as before)
// 21 August 2006: version 3.1: Prints out time info to screen only for every 20th stack now
// 21 September 2006: Fixed a few more memory bugs
// 12 July 2007: Fixed a few more memory bugs
// 7 October 2007: version 3.3: Added box counting to determine volume fraction in percolated cluster, and out of it
// 10 October 2007: Fixed more memory bugs
// 11 October 2007: Added OpenMP support


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

int main(int argc, char* argv[])
{
	clock_t starttime = clock();
	if (argc < 10) {
		cout << "Program for structural analysis of 3-D xyz data over time" << endl;
		cout << "(C)opyright 2007 Peter J. Lu" << endl;
		cout << "Version " << Version_Number << "; last updated " << Date_Last_Updated << endl << endl;
		cout << "plu_Struct3Dt.exe filestem start_stack end_stack execution radius bondlength bound corr_max_r corr_bins" << endl;
		cout << "3Dt Data file is named filestem_xyzt.txt, and all parameters are the units (e.g. microns) used in that file." << endl << endl;
		cout << "Start_stack: first stack from which to analyze data." << endl;
		cout << "End_stack: last stack from which to analyze data." << endl;
		cout << "execution: 3-bit mode of execution determines tests to run: "<<endl;
		cout << "\t least to most sig. bits: run g(r), run clusters, run fractals, run chains" << endl;
		cout << "radius: particle radius (estimate, for volume fraction calculation only)" << endl;
		cout << "bondlength: maximum inter-center separation for two particles to be considered bonded." << endl;
		cout << "cube_edge:	edge length of grid used to discretize structures for intertemporal correlation." << endl;
		cout << "corr_max_r: maximum radius for which to calculate radial distribution/correlation functions." << endl;
		cout << "corr_bins: number of bins to use in calculating radial distribution/correlation functions." << endl;
		return -1;
	}

	//determine number of stacks (and width of output arrays)
	const int start_stack = atoi(argv[2]);
	const int end_stack = atoi(argv[3]);
	const int num_stacks = end_stack - start_stack + 1;
	if(num_stacks < 1) {
		cerr << "Number of stacks is _not_ positive; enter some better parameters!" << endl;
		return -1;
	}
	const int execution_mode = atoi(argv[4]);
	if (execution_mode == 0) {
		cerr << "Executing only intertemporal correlations." << endl;
	}
	const bool run_gr3 = (execution_mode & 1);		//bitwise AND operation with 1 = binary 0001
	const bool run_cluster = (execution_mode & 2);	//bitwise AND operation with 2 = binary 0010
	const bool run_fractal = (execution_mode & 4);	//bitwise AND operation with 4 = binary 0100
	const bool run_chains = (execution_mode & 8);	//bitwise AND operation with 4 = binary 1000
	const float particle_radius = atof(argv[5]);	//just to determine volume fraction (not otherwise used)
	//for clustering/bonding/percolation determination
	const float bond_radius = atof(argv[6]);
	const float bond_r2 = bond_radius * bond_radius;
	const float perc_distance = 1.5 * bond_radius;		//particle is considered to touch the boundary if its center is within 1.5 bond radii
	//for intertemporal correlation
	const float cubegrid_edge_unitlength = atof(argv[7]);
	//for g(r) and density correlation functions
	const float max_radius = atof(argv[8]);
	const int num_bins = atoi(argv[9]);

	//create input data filename and filestream
	string filestem = argv[1];
	string infilename = filestem + "_xyzt.txt";
	ifstream indatafile;		//input data file (output from centerfinding software)
	indatafile.open(infilename.c_str(),ios::in);
	if(!indatafile) {
		cerr << "Cannot open input data file. Check file path and/or filename, and try again." << endl;
		return -1;
	}

	//log file: initial information, parameters, etc.
	string logfilename = filestem + "_struct3D_log.txt";
	ofstream logfile;		//output log file
	logfile.open(logfilename.c_str(), ios::out);
	logfile << "Program for structural analysis of 3-D xyz data over time" << endl;
	logfile << "(C)opyright 2007 Peter J. Lu" << endl;
	logfile << "Version " << Version_Number << "; last updated " << Date_Last_Updated <<  endl << endl;
	logfile << "3Dt data input file: " << infilename << endl;
	logfile << "This log file: " << logfilename << endl << endl;
	logfile << "Analyzing data from stacks " << start_stack << " to " << end_stack << "." << endl;
	cout << "Program for structural analysis of 3-D xyz data over time" << endl;
	cout << "(C)opyright 2007 Peter J. Lu" << endl;
	cout << "Version " << Version_Number << "; last updated " << Date_Last_Updated <<  endl << endl;
	if(run_gr3 == true) {
		cout << "Executing 3D g(r) calculation." << endl;
		logfile << "Executing 3D g(r) calculation." << endl;
	}
	if(run_cluster == true) {
		cout << "Executing bonding/clustering/percolation calculation." << endl;
		logfile << "Executing bonding/clustering/percolation calculation." << endl;
	}
	if(run_fractal == true) {
		cout << "Executing fractal dimension analysis using FD3; make sure FD3_plu.exe is in same directory." << endl;
		logfile << "Executing fractal dimension analysis using FD3." << endl;
	}
	if(run_chains == true) {
		cout << "Executing distance-dependent count of chains emanating from each particle." << endl;
		logfile << "Executing distance-dependent count of chains emanating from each particle." << endl;
	}
	logfile << "Particle radius: " << particle_radius << endl;
	logfile << "Two particles are considered bonded if their centers are within " << bond_radius << endl;
	logfile << "Particles less than " << perc_distance << " from boundary are considered touching (for percolation determination)." << endl;
	logfile << "g(r) calculated out to r = " << max_radius << " in " << num_bins << " bins." << endl;
	logfile << "Intertemporal correlation discretizes particle positions on a cubic grid of edge length " << cubegrid_edge_unitlength << endl;
	logfile << endl << "Columns in _stackstats.txt file: stack number; number of particles; overall volume fraction;" << endl;

	logfile << "fraction of total particles that are in a percolated cluster; volume fraction of particles within the percolated cluster;" << endl;
	logfile << "volume fraction of particles outside of the percolated cluster; fraction of total boxes that are occupied by the percolated cluster;" << endl;
	logfile << "intertemporal correlation coefficient; " << endl;
	logfile << "capacity fractal dimension; information fractal dimension; correlation fractal dimension" << endl << endl;

	//read first line of data file to figure out how many bytes it contains
	//need to back up one row, since while loop checks for differences in stack number
	//and exits, but after having read the row. So we need to check every iteration to see how many
	//bytes the last line is
	float (*stackfirstrow)[NUMCOLUMNS] = new float[1][NUMCOLUMNS];
	int p=0;
	for(p=0; p<NUMCOLUMNS; p++) {
		indatafile >> stackfirstrow[0][p];
	}
	int linesize = indatafile.tellg();
	delete [] stackfirstrow;
	stackfirstrow = NULL;

	//file creation for summary output data
	ofstream outfile_stackstats;
	string outfilename_stackstats = filestem + "_stackstats.txt";
	outfile_stackstats.open(outfilename_stackstats.c_str(),ios::out);
	outfile_stackstats.precision(4);

	//INITIALIZATION for cluster analysis routines
	//Create output files
	ofstream outfile_clusters;
	ofstream outfile_clustersizes;
	ofstream outfile_largestclusterdata;
	ofstream outfile_particles;
	//Prepare and initialize relevant arrays
	//first column of nn list, which I output, contains the numbers 1-20; I don't need to store that.
//	const int nn_array_size = num_stacks * 20;
	const int nn_array_size = num_stacks * NN_ARRAY_ELEMENTS;
	float *nn_histograms = new float[nn_array_size];
	const int clusterlogbin_array_size = NUMLOGBINS * (num_stacks + 1);
	float *clustersizes_logbins = new float[clusterlogbin_array_size];
	int num_percolated_particles = 0;	//number of particles in a stack which are part of percolated cluster(s)
	float percolated_vol_frac = 0;		//volume fraction of particles in percolated clusters from discrete grid
	float non_percolated_vol_frac = 0;	//volume fraction of particles NOT in percolated clusters, from discrete grid
	float frac_percolated = 0;			//total fraction of sample volume occupied by percoltaed clusters, from discrete grid

	if(run_cluster == true) {
		string outfilename_particles = filestem + "_particles.txt";
		outfile_particles.open(outfilename_particles.c_str(),ios::out);
		string outfilename_clusters = filestem + "_clusters.txt";
		outfile_clusters.open(outfilename_clusters.c_str(),ios::out);
		string outfilename_largestclusterdata = filestem + "_largest_cluster_data.txt";
		outfile_largestclusterdata.open(outfilename_largestclusterdata.c_str(),ios::out);
		string outfilename_clustersizes = filestem + "_cluster_sizes.txt";
		outfile_clustersizes.open(outfilename_clustersizes.c_str(),ios::out);

		for(p=0;p<nn_array_size;p++) {
			nn_histograms[p] = 0;
		}
		for(p=0;p<clusterlogbin_array_size;p++) {
			clustersizes_logbins[p] = 0;
		}
	}

	//INITIALIZATION for g(r) calculation
	//create 1-dim arrays to hold output data (since we can't dynamically allocate in two dimensions)
	//pitch is the width of the array, = number of columns: num_stacks
	//so when calling the Index_1D function, use the convention (row number, stack number, num_stacks)
	const int corr_array_size = (num_stacks + 1) * num_bins;
	float *gr3_output = new float[corr_array_size];
	float *corr_output = new float[corr_array_size];
	if(run_gr3 == true) {
		//create filenames and files for big data dump files with full analysis (error bars, etc.)
		/*
		string outfilestem_all_gr3 = filestem + "_all_gr3.txt";
		outfile_all_gr3.open(outfilestem_all_gr3.c_str(),ios::out);
		string outfilestem_all_corr = filestem + "_all_corr.txt";
		outfile_all_corr.open(outfilestem_all_corr.c_str(),ios::out);
		*/
		for(p=0;p<corr_array_size;p++) {
			gr3_output[p] = 0;
			corr_output[p] = 0;
		}
	}

	//INITIALIZATION for distance-dependent count of chains emanating from every particle
	//create 1-dim arrays to hold output data (since we can't dynamically allocate in two dimensions)
	//pitch is the width of the array, = number of columns: num_stacks
	//so when calling the Index_1D function, use the convention (row number, stack number, num_stacks)
	const int num_shells = 100;	//this just needs to be bigger than the number calculated based on # of bond-distances to nearest boundary
	const int chains_array_size = (num_stacks + 1) * num_shells;
	float *chains_output = new float[chains_array_size];
	if(run_chains == true) {
		for(p=0;p<chains_array_size;p++) {
			chains_output[p] = 0;
		}
	}


	//INITIALIZATION for calculation of correlations between subsequent stacks and z-density counts
	//discretize particle coordinates into 60 micron cubes
	const int max_linear_dimension = 60;	//change this if any dimension is greater than 60 units
	const int total_grid_edge_length = max_linear_dimension / cubegrid_edge_unitlength;	//
	const int total_volume_elements = total_grid_edge_length  * total_grid_edge_length  * total_grid_edge_length;
	bool *discrete_positions_current = new bool[total_volume_elements];		//discretized particle coordinates for current stack
	bool *discrete_positions_previous = new bool[total_volume_elements];		//discretized particle coordinates for immediately preceding stack
	clear_discrete_positions(discrete_positions_current,total_volume_elements);
	clear_discrete_positions(discrete_positions_previous,total_volume_elements);
	//create output data array for z-density (count number of particles with z-coordinate within one bin, using binsize from above)
	const int z_bin_arraysize = total_grid_edge_length * (num_stacks+1);
	int *z_counts = new int[z_bin_arraysize];
	for (p=0; p<z_bin_arraysize; p++) {
		z_counts[p] = 0;
	}
	bool firststackflag = true;	//just want a check flag so that there will be no attempt at cross correlation for the first stack
	//other variables that are used in the while loop
	int status = 1, particles_in_this_stack = 0, stacknum = 0, i=0;
	float capacity_fracdim=0, info_fracdim=0, corr_fracdim=0;

	//START OF MAIN LOOP of program: loads one stack of data, does analysis, then repeats until done.
	while(indatafile.eof() == false) {
		float (*infiledata)[NUMCOLUMNS] = new float[NUMROWS][NUMCOLUMNS];


		particles_in_this_stack = read_stack_from_file(indatafile, infiledata, linesize);

		stacknum = infiledata[0][0];
		if (stacknum <=0) {
			cerr << "Stack number " << stacknum << " is less than or equal to zero; this is wrong." << endl;
		}
		if (stacknum > end_stack) {
			break;
		}
		logfile << "Stack " << stacknum << " has " << particles_in_this_stack << " rows." << endl;
		print_row(infiledata, 0, logfile);
		print_row(infiledata, particles_in_this_stack-1, logfile);

		//Run cluster analysis
		if(stacknum >= start_stack && stacknum <= end_stack && run_cluster == true) {
			//copy over relevant data only to new array, then pass this, so original doesn't get contaminated
			float (*onestack_xyz_list)[3] = new float[particles_in_this_stack][3];
			for(i=0; i<particles_in_this_stack; i++) {
				onestack_xyz_list[i][0]=infiledata[i][1];	//x-coordinate
				onestack_xyz_list[i][1]=infiledata[i][2];	//y-coordinate
				onestack_xyz_list[i][2]=infiledata[i][3];	//z-coordinate
			}

			status = cluster_analysis(particles_in_this_stack, onestack_xyz_list, nn_histograms, clustersizes_logbins,
				num_stacks, stacknum, bond_r2, perc_distance, total_grid_edge_length, cubegrid_edge_unitlength, particle_radius,
				num_percolated_particles, percolated_vol_frac, non_percolated_vol_frac, frac_percolated,
				logfile, outfile_particles, outfile_clusters, outfile_largestclusterdata, outfile_clustersizes, run_fractal);
			if(status != 0) {
				logfile << "Error in executing cluster analysis for stack: " << stacknum << endl;
			}
			logfile << "Particles in percolated clusters: " << num_percolated_particles << endl;

			//Housekeeping: clean up memory
			delete [] onestack_xyz_list;
			onestack_xyz_list = NULL;
		}

		//Run correlation and g(r) analysis
		if(stacknum >= start_stack && stacknum <= end_stack && run_gr3 == true) {
			float (*onestack_xyz_list)[3] = new float[particles_in_this_stack][3];
			for(i=0; i<particles_in_this_stack; i++) {
				onestack_xyz_list[i][0]=infiledata[i][1];	//x-coordinate
				onestack_xyz_list[i][1]=infiledata[i][2];	//y-coordinate
				onestack_xyz_list[i][2]=infiledata[i][3];	//z-coordinate
			}

			status = gr3_corr_analysis(particles_in_this_stack, onestack_xyz_list, gr3_output, corr_output,
				num_stacks, stacknum, max_radius, num_bins, logfile); //, outfile_all_gr3, outfile_all_corr);

			if(status != 0) {
				logfile << "Error in executing g(r) analysis for stack: " << stacknum << endl;
			}
			//Housekeeping: clean up memory
			delete [] onestack_xyz_list;
			onestack_xyz_list = NULL;
		}

		//Run fractal dimension analysis
		if(stacknum >= start_stack && stacknum <= end_stack && run_fractal == true) {
			capacity_fracdim=0;
			info_fracdim=0;
			corr_fracdim=0;

			//remember that properly formatted data file for FD3 is generated within the cluster analysis program
			//execute FD3_plu.exe program, which returns only three numbers: capacity, info and correlation
			//fractal dimension estimates.
			system("FD3_plu -t temp_FD_data.txt > FD_out.txt");

			//read results back in
			ifstream FD_results;
			FD_results.open("FD_out.txt",ios::in);
			FD_results >> capacity_fracdim;
			FD_results >> info_fracdim;
			FD_results >> corr_fracdim;
			FD_results.close();
			system("del FD_out.txt");
		}

		//Run chain analysis
		if(stacknum >= start_stack && stacknum <= end_stack && run_chains == true) {
			float (*onestack_xyz_list)[3] = new float[particles_in_this_stack][3];
			for(i=0; i<particles_in_this_stack; i++) {
				onestack_xyz_list[i][0]=infiledata[i][1];	//x-coordinate
				onestack_xyz_list[i][1]=infiledata[i][2];	//y-coordinate
				onestack_xyz_list[i][2]=infiledata[i][3];	//z-coordinate
			}
			//status = chains_analysis(particles_in_this_stack, onestack_xyz_list, stacknum, bond_radius, logfile, outfile_chains);
			status = chains_analysis(particles_in_this_stack, onestack_xyz_list, chains_output, num_stacks, stacknum, num_shells, bond_radius, logfile);

			if(status != 0) {
				logfile << "Error in executing chains analysis for stack: " << stacknum << endl;
			}
			//Housekeeping: clean up memory

			delete [] onestack_xyz_list;
			onestack_xyz_list = NULL;
		}

		//Run intertemporal structure correlation analysis
		if(stacknum >= start_stack && stacknum <= end_stack) {
			float (*onestack_xyz_list)[3] = new float[particles_in_this_stack][3];
			for(i=0; i<particles_in_this_stack; i++) {
				onestack_xyz_list[i][0]=infiledata[i][1];	//x-coordinate
				onestack_xyz_list[i][1]=infiledata[i][2];	//y-coordinate
				onestack_xyz_list[i][2]=infiledata[i][3];	//z-coordinate
			}

			//calculate correlation between subsequent stack discretized densities
			clear_discrete_positions(discrete_positions_current,total_volume_elements);
			discretize_positions(onestack_xyz_list, discrete_positions_current, particles_in_this_stack, total_grid_edge_length, cubegrid_edge_unitlength, particle_radius);
			float corr_coeff = cross_correlation(discrete_positions_current, discrete_positions_previous,total_volume_elements);
			//copy "current" stack data to "previous" stack data, for comparison in next iteration of while loop
			copy_discrete_positions(discrete_positions_current, discrete_positions_previous,total_volume_elements);
			total_z_counts(discrete_positions_current,z_counts,num_stacks,stacknum,total_grid_edge_length);

			//output stacknumber, volume fraction, intertemporal density correlation, and (if selected) fractal dimension (capacity, info and correlation)
			float total_system_volume = systemvolume(onestack_xyz_list,particles_in_this_stack);
			float volume_per_particle = 4 * 3.14159265 * particle_radius * particle_radius * particle_radius / 3;
			float volumefraction = volume_per_particle * particles_in_this_stack / total_system_volume;
			outfile_stackstats << stacknum << "\t" << particles_in_this_stack << "\t" << volumefraction << "\t";
			if(run_cluster == true) {
				float perc_fraction = (float) num_percolated_particles / particles_in_this_stack;
				outfile_stackstats	<< perc_fraction << "\t";				//fraction of total particles that are in a percolated cluster
				outfile_stackstats	<< percolated_vol_frac << "\t";			//volume fraction of particles within the percolated cluster
				outfile_stackstats	<< non_percolated_vol_frac << "\t";		//volume fraction of particles outside of the percolated cluster
				outfile_stackstats	<< frac_percolated  << "\t";			//fraction of total boxes that are occupied by the percolated cluster
			}
			outfile_stackstats << corr_coeff;
			if(run_fractal == true) {
				outfile_stackstats << "\t" << capacity_fracdim;
				outfile_stackstats << "\t" << info_fracdim;
				outfile_stackstats << "\t" << corr_fracdim;
			}
			outfile_stackstats << endl;

			//Housekeeping: clean up memory
			delete [] onestack_xyz_list;
			onestack_xyz_list = NULL;
		}

		//Housekeeping: clean up memory
		delete [] infiledata;
		infiledata=NULL;

		if(stacknum % 20 == 1) {
			cout << "Elapsed time after analyzing stack " << stacknum << ": " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl;
		}
		logfile << "Elapsed time after linking stack " << stacknum << ": " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl << endl;
	}
	//END OF MAIN LOOP of program.


	//print out data files
	if(run_cluster == true) {
		ofstream outfile_nn;
		string outfilename_nn = filestem + "_nn_hist.txt";
		outfile_nn.open(outfilename_nn.c_str(),ios::out);
		int row=0;
		for(row=0; row < 20; row++) {
			outfile_nn << row << "\t";
			for(int column = 0; column <num_stacks; column++) {
				outfile_nn << nn_histograms[Index_1D(row,column,num_stacks)] << "\t";
			}
			outfile_nn << endl;
		}
		outfile_nn.close();

		ofstream outfile_logbinclustersizes;
		string outfilename_logbinclustersizes = filestem + "_clustersizes_logdist.txt";
		outfile_logbinclustersizes.open(outfilename_logbinclustersizes.c_str(),ios::out);
		for(row=0; row<NUMLOGBINS-1; row++) {
			for(int column = 0; column < num_stacks+1; column++) {
				outfile_logbinclustersizes << clustersizes_logbins[Index_1D(row,column,num_stacks+1)] << "\t";
			}
			outfile_logbinclustersizes << endl;
		}
		outfile_logbinclustersizes.close();
		outfile_clusters.close();
		outfile_clustersizes.close();
		outfile_largestclusterdata.close();
		outfile_particles.close();

	}

	if(run_gr3 == true) {
		//create filenames and files for graphable output files
		//first column is x-axis (r), second to nth columns are g(r)s for different stacks
		ofstream outfile_gr3;
		ofstream outfile_corr;
		string outfilestem_gr3 = filestem + "_gr3.txt";
		outfile_gr3.open(outfilestem_gr3.c_str(),ios::out);
		outfile_gr3.precision(4);
		string outfilestem_corr = filestem + "_corr.txt";
		outfile_corr.open(outfilestem_corr.c_str(),ios::out);
		for(int row=0; row<num_bins; row++) {
			for(int column = 0; column <=num_stacks; column++) {
				outfile_gr3 << gr3_output[Index_1D(row,column,num_stacks+1)] << "\t";
				outfile_corr << corr_output[Index_1D(row,column,num_stacks+1)] << "\t";
			}
			outfile_gr3 << endl;
			outfile_corr << endl;
		}
		outfile_gr3.close();
		outfile_corr.close();
	}

	if(run_chains == true) {
		ofstream outfile_chains;
		string outfilename_chains = filestem + "_chains.txt";
		outfile_chains.open(outfilename_chains.c_str(),ios::out);
		for(int row=0; row<num_shells; row++) {
			if(chains_output[Index_1D(row,0,num_stacks+1)] > EPSILON) {
				for(int column = 0; column <=num_stacks; column++) {
					outfile_chains << chains_output[Index_1D(row,column,num_stacks+1)] << "\t";
				}
				outfile_chains << endl;
			}
		}
		outfile_chains.close();
	}

	ofstream outfile_zcounts;
	string outfilename_zcounts = filestem + "_zcounts.txt";
	outfile_zcounts.open(outfilename_zcounts.c_str(),ios::out);
	for(int q=0;q<total_grid_edge_length;q++){
		outfile_zcounts << cubegrid_edge_unitlength * q;
		for(p=1;p<num_stacks+1;p++){
			outfile_zcounts  << "\t" << z_counts[Index_1D(q,p,num_stacks+1)];
		}
		outfile_zcounts << endl;
	}
	outfile_zcounts.close();

	//cleanup/close input and summary output data files
	indatafile.close();
	outfile_stackstats.close();

	//cleanup memory
//	delete [] discrete_positions_previous;
//	discrete_positions_previous = NULL;
//	delete [] discrete_positions_current;
//	discrete_positions_current = NULL;

	delete [] clustersizes_logbins;
	clustersizes_logbins = NULL;
	delete [] nn_histograms;
	nn_histograms = NULL;

	delete [] gr3_output;
	gr3_output = NULL;
	delete [] corr_output;
	corr_output = NULL;

	delete [] chains_output;
	chains_output = NULL;

	delete [] z_counts;
	z_counts = NULL;


	//Show total time to run program
	cout << "Total time: " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl;
	logfile << "Total time: " << (float) (clock() - starttime)/CLOCKS_PER_SEC << " seconds" << endl;
	logfile.close();
	return 0;
}
