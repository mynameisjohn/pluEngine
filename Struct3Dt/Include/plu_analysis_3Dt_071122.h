#ifndef PLU_ANALYSIS_3DT1_H
#define PLU_ANALYSIS_3DT1_H

#include <fstream>
#define NUMCOLUMNS 4	//number of colums in INPUT data file
#define NUMLOGBINS 51	//number of logarithmically-spaced bins used in cluster size distributions
#define NN_ARRAY_ELEMENTS 50
#define MAXROWS 10000


int Index_1D(const int row, const int column, const int pitch);		//general function for returning the 1-D index of a 2-D array

int Index_3Dcube_to_1D(const int x, const int y, const int z, const int cubeedgelength);	//function for returning a 1D index from a 3D "array" that is a cube

void print_row(float (*data)[NUMCOLUMNS], const int row, ostream &out);

int smallest(const int index1, const int index2, const int index3);	//determines which of the three integers are smallest (for indexing)

float smallestf(const float num1, const float num2, const float num3);

float smaller(const float num1, const float num2);

int cluster_analysis(const int totalnumparticles, float (*positionlist)[3], float *nn_histograms, float *logbin_clustersizes,
					 const int totalstacknumber, const int currentstacknumber, const float bond_r2, const float perc_distance,
					 const int num_elements_on_grid_edge, const float cubegrid_edge_unitlength, const float particle_radius,
					 int &num_perc_particles, float &perc_vol_frac, float &non_perc_vol_frac, float &frac_percolated,
					 ofstream &logfile, ofstream &particledatafile, ofstream &clusterdatafile, ofstream &largestclusterdatafile,
					 ofstream &clustersize_countlist, const bool run_fractal);		//main function for running the cluster analysis
					//perc_vol_frac: volume fraction of particles in percolated clusters
					//non_perc_vol_frac: volume fraction of particles not in percolated clusters
					//frac_percolated: fraction of total sample volume that is occupied by percolated cluster

int neighborcount_3D(const bool (*discrete_positions), int i, int j, int k, const int num_elements_on_grid_edge);

int gr3_corr_analysis(const int totalnumparticles, float (*positionlist)[3], float *gr3_output, float *corr_output, 
					  const int totalstacknumber, const int currentstacknumber,  const float max_radius_gr3, const int num_bins, 
					  ofstream &logfile); //, ofstream &gr3datafile, ofstream &corrdatafile);				

void discretize_positions(const float (*positionlist)[3], bool *discrete_positions, const int total_particles, const int totalgrid_edgelength, const float edge_unitlength, const float particle_radius);

float cubegrid_occupied_volume(const float (*positionlist)[3], const int totalnumparticles, const float x_origin, const float y_origin, const float z_origin, 
//float cubegrid_occupied_volume(float positionlist[MAXROWS][3], const int totalnumparticles, const float x_origin, const float y_origin, const float z_origin, 
							   const float cube_edge_length, const float particle_radius, const float bond_distance);

float delta_r_2D(const float x1, const float x2, const float y1, const float y2);

float delta_r_3D(const float x1, const float x2, const float y1, const float y2, const float z1, const float z2);

void copy_discrete_positions(bool *discrete_positions_source, bool *discrete_positions_target, const int total_volume_elements);

void clear_discrete_positions(bool *discrete_positions, const int total_volume_elements);

void total_z_counts(bool *discrete_positions, int *z_counts, const int totalstacknumber, const int currentstacknumber, 
					const int totalgrid_edgelength);

float cross_correlation(bool *discrete_positions1, bool *discrete_positions2, const int total_volume_elements);

float systemvolume(float (*positionlist)[3], const int total_particles);

int num_clusters_in_shell(const int totalnumparticles, float (*positionlist)[3], const float bond_r2);

int chains_analysis(const int totalnumparticles, float (*positionlist)[3], float *chains_output,
					const int totalstacknumber, const int currentstacknumber, const int total_number_of_shells,
					const float shell_width, ofstream &logfile);

#endif