#include <ctime>
#include <string>
using namespace std;
#include <fstream>
#include <iostream>
#include <cmath>
#include "plu_analysis_3Dt_071122.h"
#define PI 3.14159265
//#include <omp.h>
//Modification History
//Version 1.0: calculates nearest neighbors, their distribution, cluster identity, their size distribution (March, 2005)
//**decided to get rid of bond anisotropy vector, since the way it's calculated just doesn't seem to work at all (May 5, 2005)
//Version 2.0: added logarithmic binning, cluster radius of gyration (began May 5, 2005)
//Version 2.1: added average Rg for each cluster mass (June 5, 2005)
//Version 2.2: fixed "bug" in which calculated delta_r^2 was an integer, not a float
//Version 2.3: reimplemented cluster label counting to be more efficient
//Version 2.4: Integrated into multiple-stack framework of Struct3Dt: August 28, 2005
//Version 2.5: Added output of line from largest cluster to separate text file, for real-time target locking
//Version 3.0: Added determination of distinct number of chains as a function of distance from each point: June 4, 2006
//Version 3.1: Fixed bug so that 9th column of particle data now properly shows Rg (not Rg^2, as before)
//Version 3.3: Added volume fraction calculation within percolated clusters, via box counting, 07 October 2007
//Version 3.4: Added volume fraction calculation, defined as volume excluded to a test-particle the size of the regular particles
//Version 3.5: Removed volume fraction calculation with subgrid (only large grid remains), now residing in plu_part2RIB_gel_block.exe code, 22 November 2007
//(C)opyright 2007 Peter J. Lu. All Rights Reserved. Last Modified, 11 November 2007

int Index_1D(const int row, const int column, const int pitch) {
	//converts 2-D array address into a 1-D number
	//'pitch' is the total number of columns; i.e. the width
	//'column' ranges from 0 to pitch-1
	//'row' ranges from 0 to whatever
	return column + row * pitch;
}

void print_row(float (*data)[NUMCOLUMNS], const int row, ostream &out) {
	for(int i=0;i<NUMCOLUMNS;i++) {
		out << data[row][i] << "\t";
	}
	out << endl;
}

int smallest(const int index1, const int index2, const int index3) {
	int value = 0;
	if(index1 <= index2 && index1 <= index3) {
		value = index1;
	}
	else if(index2 <= index3 && index2 <= index1){
		value = index2;
	}
	else if(index3 <= index2 && index3 <= index1){
		value = index3;
	}
	return value;
}

float smallestf(const float num1, const float num2, const float num3) {
	float value = 0;
	if(fabs(num1) <= fabs(num2) && fabs(num1) <= fabs(num3)) {
		value = fabs(num1);
	}
	else if(fabs(num2) <= fabs(num3) && fabs(num2) <= fabs(num1)) {
		value = fabs(num2);
	}
	else if(fabs(num3) <= fabs(num2) && fabs(num3) <= fabs(num1)) {
		value = fabs(num3);
	}
	return value;
}


float smaller(const float num1, const float num2) {
	float value = 0;
	//note that this function checks the absolute values of index1 and index2
	if(fabs(num1) < fabs(num2)) {
		value = fabs(num1);
	}
	else {
		value = fabs(num2);
	}
	return value;
}

int cluster_analysis(const int totalnumparticles, float (*positionlist)[3],  float *nn_histograms, float *logbin_clustersizes,
					 const int totalstacknumber, const int currentstacknumber, const float bond_r2, const float perc_distance,
					 const int num_elements_on_grid_edge, const float cubegrid_edge_unitlength, const float particle_radius,
					 int &num_perc_particles, float &perc_vol_frac, float &non_perc_vol_frac, float &frac_percolated,
					 ofstream &logfile, ofstream &particledatafile, ofstream &clusterdatafile, ofstream &largestclusterdatafile,
					 ofstream &clustersize_countlist, const bool run_fractal)
{

	//1. calculate maximum values for extents
	int j = 0;
	float minx = 1000, maxx = 0, miny = 1000, maxy = 0, minz = 1000, maxz = 0;

	for (j=0; j < totalnumparticles; j++) {
		if(positionlist[j][0] < minx) {
			minx = positionlist[j][0];
		}
		if(positionlist[j][1] < miny) {
			miny = positionlist[j][1];
		}
		if(positionlist[j][2] < minz) {
			minz = positionlist[j][2];
		}
		if(positionlist[j][0] > maxx) {
			maxx = positionlist[j][0];
		}
		if(positionlist[j][1] > maxy) {
			maxy = positionlist[j][1];
		}
		if(positionlist[j][2] > maxz) {
			maxz = positionlist[j][2];
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
	logfile << "minimum x: " << minx << "\tminimum y: " << miny << "\tminimum z: " << minz << endl;
	logfile << "maximum x: " << maxx << "\tmaximum y: " << maxy << "\tmaximum z: " << maxz << endl;

	logfile << "Extent of system in x,y,z: " << deltax << ", " << deltay << ", " << deltaz << endl;

	//2. Create data structure of particles for analysis
	//column 0: x-position (microns)
	//column 1: y-position (microns)
	//column 2: z-position (microns)
	//column 3:	cluster label (raw)
	//column 4: number of nearest neighbors
	//column 5:	cluster label (consolidated)
	//column 6: delta_x vector = x displacement relative to cluster center
	//column 7: delta_y vector = y displacement relative to cluster center
	//column 8: delta_z vector = z displacement relative to cluster center
	//column 9: r2 displacement = (delta_x^2 + delta_y^2 + delta_z^2) relative to cluster center
	float (*particle_list)[10] = new float[totalnumparticles][10];

	//create array to hold index values, which will then be sorted to backpropagate to particle list
	int *cluster_index_list = new int[totalnumparticles];

	//3. Copy particle data, assign cluster number
	for(j = 0; j < totalnumparticles; j++) {
		particle_list[j][0] = positionlist[j][0];	//copy x-position data
		particle_list[j][1] = positionlist[j][1];	//copy y-position data
		particle_list[j][2] = positionlist[j][2];	//copy z-position data
		for(int l = 3; l<10; l++) {
			particle_list[j][l] = 0;
		}
		cluster_index_list[j] = j;
	}

	//4. Loop through particle list to determine bonding, count nearest neighbors
	int i=0, k=0, oldsmallerindex = 0, oldsmallerindex2 = 0, newsmallerindex=0;
	float r2 = 0;
	float delta_x = 0, delta_y = 0, delta_z = 0;

	for(i=0; i<totalnumparticles-1; i++) {
		for(j=i+1; j<totalnumparticles; j++) {
			//calculates radial distance between the two particles = bond distance
			r2 = (particle_list[i][0]-particle_list[j][0]) * (particle_list[i][0]-particle_list[j][0]) +
				(particle_list[i][1]-particle_list[j][1]) * (particle_list[i][1]-particle_list[j][1]) +
				(particle_list[i][2]-particle_list[j][2]) * (particle_list[i][2]-particle_list[j][2]);

			//check if bond is within threshold
			if(r2 < bond_r2) {
				//increase nearest neighbor count
				particle_list[i][4]++;
				particle_list[j][4]++;
				//determine if this is first bond being made (no reassignment necessary)
				if(cluster_index_list[j] == j) {
					cluster_index_list[j]=cluster_index_list[i];
				}
				//otherwise, need to propagate the labels
				else {
					oldsmallerindex = cluster_index_list[j];
					oldsmallerindex2 = cluster_index_list[i];
					newsmallerindex = smallest(oldsmallerindex,oldsmallerindex2,cluster_index_list[oldsmallerindex]);
					for(k=0; k<totalnumparticles; k++) {
						if(cluster_index_list[k]==oldsmallerindex || cluster_index_list[k]==oldsmallerindex2) {
							cluster_index_list[k]=newsmallerindex;
						}
					}
				}
			}
		}
	}
	//5. Assign cluster labels to particle data
	for(k=0; k<totalnumparticles; k++) {
		particle_list[k][3] = cluster_index_list[k];
	}
	delete [] cluster_index_list;
	cluster_index_list = NULL;

	//6. Create new cluster array, sorting size and percolation information
	//column 0: number of particles in that cluster/particle count (i.e. size)
	//column 1: boolean: does the cluster have a particle within perc_distance of minimum x-value?
	//column 2: boolean: does the cluster have a particle within perc_distance of maximum x-value?
	//column 3: boolean: does the cluster have a particle within perc_distance of minimum y-value?
	//column 4: boolean: does the cluster have a particle within perc_distance of maximum y-value?
	//column 5: boolean: does the cluster have a particle within perc_distance of minimum z-value?
	//column 6: boolean: does the cluster have a particle within perc_distance of maximum z-value?
	//column 7: consolidated cluster label (after stripping out zeros)
	//column 8: cluster x-center position
	//column 9: cluster y-center position
	//column 10: cluster z-center position
	//column 11: cluster radius of Gyration Rg
	//column 12: boolean: is cluster percolated?
	float (*cluster_list)[13] = new float[totalnumparticles+1][13];

	//this array mostly empty array because the cluster numbers are not sorted.
	//row number is the cluster label

	//fill with zeros (minimum cluster label should be zero, from above)
	for(i=0; i<=totalnumparticles; i++) {
		for(j=0;j<13;j++) {
			cluster_list[i][j] =0;
		}
	}

	//7. pass through particle list to count particles in each cluster, and determine percolation
	int cluster_label = 0;
	for(i=0;i<totalnumparticles;i++) {
		//increment particle count
		cluster_label = (int) particle_list[i][3];
		cluster_list[cluster_label][0]++;

		//bounds checking for percolation: set boolean values
		if( (particle_list[i][0] - minx) < perc_distance) {
			cluster_list[cluster_label][1] = 1;
		}
		if( (maxx - particle_list[i][0]) < perc_distance) {
			cluster_list[cluster_label][2] = 1;
		}

		if( (particle_list[i][1] - miny) < perc_distance) {
			cluster_list[cluster_label][3] = 1;
		}
		if( (maxy - particle_list[i][1]) < perc_distance) {
			cluster_list[cluster_label][4] = 1;
		}

		if( (particle_list[i][2] - minz) < perc_distance) {
			cluster_list[cluster_label][5] = 1;
		}
		if( (maxz - particle_list[i][2]) < perc_distance) {
			cluster_list[cluster_label][6] = 1;
		}
	}

	//8. Consistency checks: see if all monomers belongs to a cluster of size 1
	int errorcount1 = 0, errorcount2 = 0;
	for(i=0;i<totalnumparticles;i++) {
		cluster_label = (int) particle_list[i][3];
		//find particles with officially-found neighbors, but yet are registering as part of clusters of size 1
		if( (int) particle_list[i][4] != 0 && (int) cluster_list[cluster_label][0]==1 ) {
			if (errorcount1 == 0) {
				cout << "The following particles have >0 nearest neighbors, but are clusters of size 1" << endl;
				logfile << "The following particles have >0 nearest neighbors, but are clusters of size 1" << endl;
			}
			cout << cluster_label << "\t";
			logfile << cluster_label << "\t";
			errorcount1++;
		}
	}
	for(i=0;i<totalnumparticles;i++) {
		cluster_label = (int) particle_list[i][3];
			if( (int) particle_list[i][4] == 0 && (int) cluster_list[cluster_label][0]!=1) {
			if (errorcount2 == 0) {
				cout << "\nThe following particles have 0 nearest neighbors, but are clusters of size >1" << endl;
				logfile << "\nThe following particles  have 0 nearest neighbors, but are clusters of size >1" << endl;
			}
			cout << cluster_label << "\t";
			logfile << cluster_label << "\t";
			errorcount2++;
		}
	}
	logfile << endl << "Total monomer mismatches (should be zero): " << errorcount1 + errorcount2 << endl;

	//9. Calculate cluster center and Radius of Gyration Rg
	//add x, y, and z positions of each particle to the appropriate cluster
	for(i=0;i<totalnumparticles;i++) {
		cluster_label = (int) particle_list[i][3];
		cluster_list[cluster_label][8]+=particle_list[i][0];
		cluster_list[cluster_label][9]+=particle_list[i][1];
		cluster_list[cluster_label][10]+=particle_list[i][2];
	}
	//divide total x, y and z displacements by number of particles to locate the center for each cluster
	for(i = 0; i<=totalnumparticles; i++) {
		if( (int) cluster_list[i][0] !=0) {
			cluster_list[i][8] /= cluster_list[i][0];	//x-center
			cluster_list[i][9] /= cluster_list[i][0];	//y-center
			cluster_list[i][10] /= cluster_list[i][0];	//z-center
		}
	}
	//subtract cluster center from each particle to get relative displacement,
	//calculate squared magnitude R2, and add to cluster
	delta_x =0, delta_y=0, delta_z=0;
	float R2=0;
	for(i=0;i<totalnumparticles;i++) {
		cluster_label = (int) particle_list[i][3];

		delta_x = particle_list[i][0] - cluster_list[cluster_label][8];		//relative x
		delta_y = particle_list[i][1] - cluster_list[cluster_label][9];		//relative y
		delta_z = particle_list[i][2] - cluster_list[cluster_label][10];	//relative z
		R2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

		particle_list[i][6] = delta_x;
		particle_list[i][7] = delta_y;
		particle_list[i][8] = delta_z;
		particle_list[i][9] = R2;
		cluster_list[cluster_label][11] += R2;
	}
	//normalize total R2 by the number of particles to get Rg for each cluster
	for(i = 0; i<=totalnumparticles; i++) {
		if( (int) cluster_list[i][0] !=0) {
			cluster_list[i][11] /= cluster_list[i][0];
		}
	}

	//10. Consolidate cluster labels
	int count = 1, raw_cluster_number = 0, consolidated_cluster_number = 0;
	for(i = 0; i<=totalnumparticles; i++) {
		if( (int) cluster_list[i][0] !=0) {
			cluster_list[i][7]=count++;
		}
	}
	for(i = 0; i<totalnumparticles; i++) {
		raw_cluster_number = particle_list[i][3];
		consolidated_cluster_number = cluster_list[raw_cluster_number][7];
		particle_list[i][5] = consolidated_cluster_number;
	}
	logfile << "Total number of clusters: " << --count << endl;

	//11. determine if cluster is percolated, and write output: cluster file
	int largest_cluster = 0, largest_cluster_index = 0;
	num_perc_particles = 0;
	for(i = 0; i<=totalnumparticles; i++) {
		if( (int) cluster_list[i][0] !=0) {
			//write cluster label and size
			//cout  << count << '\t' << cluster_list[i][0] << endl;
			clusterdatafile << currentstacknumber << '\t';	//stack number
			clusterdatafile << cluster_list[i][7] << '\t';	//cluster label (consolidated)
			clusterdatafile << cluster_list[i][0] << '\t';	//cluster mass (number of particles)
			clusterdatafile << cluster_list[i][8] << '\t';	//x-center
			clusterdatafile << cluster_list[i][9] << '\t';	//y-center
			clusterdatafile << cluster_list[i][10] << '\t';	//z-center
			clusterdatafile << sqrt(cluster_list[i][11]) << '\t';	//Rg

			//determine if percolated, write 1 if so, and add to tally of particles in percolated clusters
			if( ((int) cluster_list[i][1] == 1 && (int) cluster_list[i][2] == 1) ||
				((int) cluster_list[i][3] == 1 && (int) cluster_list[i][4] == 1) ||
				((int) cluster_list[i][5] == 1 && (int) cluster_list[i][6] == 1) ) {
				logfile << "Cluster " << cluster_list[i][7] << " is percolated!" << endl;
				clusterdatafile << "1";
				cluster_list[i][12] = 1;
				num_perc_particles += cluster_list[i][0];
			}
			else {
				clusterdatafile << "0";
			}
			clusterdatafile << endl;

			//determine largest cluster
			if(cluster_list[i][0] > largest_cluster) {
				largest_cluster = cluster_list[i][0];
				largest_cluster_index = i;
			}
		}
	}
	logfile << "Largest cluster size: " << largest_cluster << endl;

	//write cluster data for largest cluster to special file
	largestclusterdatafile << currentstacknumber << '\t';	//stack number
	largestclusterdatafile << cluster_list[largest_cluster_index][7] << '\t';	//cluster label (consolidated)
	largestclusterdatafile << cluster_list[largest_cluster_index][0] << '\t';	//cluster mass (number of particles)
	largestclusterdatafile << cluster_list[largest_cluster_index][8] << '\t';	//x-center
	largestclusterdatafile << cluster_list[largest_cluster_index][9] << '\t';	//y-center
	largestclusterdatafile << cluster_list[largest_cluster_index][10] << '\t';	//z-center
	largestclusterdatafile << sqrt(cluster_list[largest_cluster_index][11]) << '\t';	//Rg
	if(int(cluster_list[largest_cluster_index][12]) == 1) {
		largestclusterdatafile << "1";
	}
	else {
		largestclusterdatafile << "0";
	}
	largestclusterdatafile << endl;

	//12. write output: particle file
	//columns: x, y, z, number of nearest neighbors, cluster number, cluster size
	int c_index = 0;
	for(i=0;i<totalnumparticles;i++) {
		particledatafile << currentstacknumber << '\t';		//stack number
		particledatafile << particle_list[i][0] << '\t';		//x-position
		particledatafile << particle_list[i][1] << '\t';		//y-position
		particledatafile << particle_list[i][2] << '\t';		//z-position
		particledatafile << particle_list[i][4] << '\t';		//number of nearest neighbors
		particledatafile << particle_list[i][9] << '\t';		//total R2 distance from cluster center
		particledatafile << particle_list[i][5] << '\t';		//cluster label (consolidated)
		c_index = (int) particle_list[i][3];
		particledatafile << cluster_list[c_index][0] << '\t';	//cluster size
		particledatafile << sqrt(cluster_list[c_index][11]) << '\t';	//cluster Rg
		particledatafile << cluster_list[c_index][12];			//boolean: is particle part of a percolated cluster
		particledatafile << endl;
	}

	//12b. Prepare properly formatted data file for FD3 program
	if(run_fractal == true) {
		ofstream temp_FD_datafile;
		temp_FD_datafile.open("temp_FD_data.txt",ios::out);
		temp_FD_datafile << num_perc_particles << endl;
		for(i=0; i<totalnumparticles; i++) {
			c_index = (int) particle_list[i][3];
			if((int) cluster_list[c_index][12] == 1) {
				temp_FD_datafile << particle_list[i][0] << "\t";
				temp_FD_datafile << particle_list[i][1] << "\t";
				temp_FD_datafile << particle_list[i][2] << endl;
			}
		}
		temp_FD_datafile.close();
	}

	//13. write output: cluster size distribution and average Rg for each size
	int *cluster_size_list = new int[largest_cluster+1];
	float *Rg_avg_list = new float[largest_cluster+1];
	int cluster_size = 0;
	for(i =0; i <=largest_cluster; i++) {
		cluster_size_list[i]=0;		//initialize array to zero
		Rg_avg_list[i]=0;		//initialize array to zero
	}


	//count clusters of a given size (assign size to array index); sparse array
	//add total Rg for all clusters of a given size (divide out later to get average)
	for(i = 0; i<=totalnumparticles; i++) {
		cluster_size = cluster_list[i][0];
		cluster_size_list[cluster_size]++;
		Rg_avg_list[cluster_size] += sqrt(cluster_list[i][11]);
	}

	//write data to file
	//divide out list of Rg's by number of clusters to get average
	for(i =1; i <=largest_cluster; i++) {
		if(cluster_size_list[i] !=0) {
			clustersize_countlist << currentstacknumber << '\t';
			clustersize_countlist << i << '\t' << cluster_size_list[i] << '\t';
			clustersize_countlist << Rg_avg_list[i]/cluster_size_list[i] << endl;
		}
	}
	delete [] Rg_avg_list;
	Rg_avg_list = NULL;

	//14. Logarithmic binning of cluster size distribution
	//51 bins from 0 to 10^5. First 10 are just numbers (0..9), then 10^1.0 to 10^5.0 in exponent tenths
	//Multiplied by 100 in order to get two decimal places for smaller bin sizes
	//const int numlogbins = 51;
	int (*logbins)[2] = new int[NUMLOGBINS][2];
	float tenpower = 0;
	for(i=0; i<10; i++) {
		logbins[i][0] = 100 * i;		//stores logarithmically binned cluster size
		logbins[i][1] = 0;				//stores number of clusters within that size range
	}

	for(i=10; i<NUMLOGBINS; i++) {
		tenpower = (float) i/10;
		logbins[i][0] = (int) 100 * pow(10,tenpower);
		logbins[i][1] = 0;
	}
	const int biggest_cluster_size = 10000000;
	//copy over cluster size list to one where index is x100.
	int *num_clusters_x100 = new int[biggest_cluster_size];
	for(i=0;i<largest_cluster;i++) {
		num_clusters_x100[100*i]=cluster_size_list[i];
	}
	//sum up bins
	for(i=0; i<NUMLOGBINS-1; i++) {
		for(j=logbins[i][0]; j<logbins[i+1][0]; j++) {
			logbins[i][1]+=num_clusters_x100[j];
		}
	}
	//copy to big data array
	for(i=0;i<NUMLOGBINS-2; i++) {
		logbin_clustersizes[Index_1D(i,0,totalstacknumber+1)]= (float) logbins[i+1][0] / 100;
		logbin_clustersizes[Index_1D(i,currentstacknumber,totalstacknumber+1)]= (float) logbins[i+1][1] * 100 / (logbins[i+2][0] - logbins[i+1][0]);
	}
	i=NUMLOGBINS-2;
	//cleanup memory
	logbin_clustersizes[Index_1D(i,0,totalstacknumber+1)]= (float) logbins[i+1][0] / 100;
	delete [] num_clusters_x100;
	num_clusters_x100 = NULL;
	delete [] logbins;
	logbins = NULL;
	delete [] cluster_size_list;
	cluster_size_list = NULL;



	//15. Determine nearest-neighbor distribution
	int largest_num_nn = 0, num_nn=0;
	largest_num_nn = NN_ARRAY_ELEMENTS;
	int *nn_list = new int[largest_num_nn+1];
	for (j=0; j<=largest_num_nn; j++) {
		nn_list[j]=0;
	}
	//calculate histogram of number of nearest neighbor bonds
	for (i=0; i<totalnumparticles; i++) {
		num_nn = particle_list[i][4];
		if(num_nn < largest_num_nn) {
			nn_list[num_nn]++;
		}
		else{
			nn_list[largest_num_nn]++;
		}
	}

	for (j=0; j<largest_num_nn; j++) {
		nn_histograms[Index_1D(j,currentstacknumber-1,totalstacknumber)] = nn_list[j];
	}
	delete [] nn_list;
	nn_list = NULL;

	//16. Determine volume fractions in percolated cluster, and for remaining particles
	if(totalnumparticles >= num_perc_particles) {
		float non_perc_particles = totalnumparticles - num_perc_particles;
		float (*perc_cluster_particles)[3] = new float[num_perc_particles+1][3];

		//add random offset to test importance of position of discrete grid
		//use even distribution, with resolution to two decimal places, up to cubegrid_edge_unitlength
		//initialize random seed
/*		srand(time(NULL));
		const float x_rand_offset = cubegrid_edge_unitlength * (rand() % 100) / 100;
		const float y_rand_offset = cubegrid_edge_unitlength * (rand() % 100) / 100;
		const float z_rand_offset = cubegrid_edge_unitlength * (rand() % 100) / 100;
		cout << "Random offsets: (" << x_rand_offset << ", " << y_rand_offset << ", " << z_rand_offset << ")"  << endl;
*/
		//copy xyz data from particle_list for particles that are part of a percolated cluster
		i=0;
		int clust_index = 0;
		for (j=0; j<totalnumparticles; j++) {
			clust_index = (int) particle_list[j][3];
			if(cluster_list[clust_index][12] == 1) {
/*				perc_cluster_particles[i][0] = particle_list[j][0] + x_rand_offset;		//x-position
				perc_cluster_particles[i][1] = particle_list[j][1] + y_rand_offset;		//y-position
				perc_cluster_particles[i][2] = particle_list[j][2] + z_rand_offset;		//z-position
*/				perc_cluster_particles[i][0] = particle_list[j][0];		//x-position
				perc_cluster_particles[i][1] = particle_list[j][1];		//y-position
				perc_cluster_particles[i][2] = particle_list[j][2];		//z-position
				i++;
			}
		}

		//calculate free volume available to a test particle, that does not overlap with any of the percolated cluster

		//discretize data from the percolated cluster only, for a large grid
		const int total_volume_elements = (num_elements_on_grid_edge+1) * (num_elements_on_grid_edge+1) * (num_elements_on_grid_edge+1);
		bool *discrete_positions = new bool[total_volume_elements];
		for(i=0;i<total_volume_elements;i++) {
			discrete_positions[i]=false;
		}

		//run simple check on whether a grid cube contains any particles, to quickly identify empty boxes, which can subsequently be ignored
		discretize_positions(perc_cluster_particles, discrete_positions, num_perc_particles, num_elements_on_grid_edge, cubegrid_edge_unitlength, 0.55*sqrt(bond_r2));

		//count up the empty boxes
		//calculate volume fractions
		int boxes_occupied_by_perc_cluster = 0, boxes_empty =0;
		for(i=0; i<total_volume_elements; i++) {
			if(discrete_positions[i] == true) {
				boxes_occupied_by_perc_cluster++;
			}
			else {
				boxes_empty++;
			}
		}
		if(boxes_occupied_by_perc_cluster + boxes_empty != total_volume_elements) {
			logfile << "Box counting error!" << endl;
		}

		//
		const float single_particle_volume = 4 * PI * particle_radius * particle_radius * particle_radius / 3;
		const float single_gridcube_volume = cubegrid_edge_unitlength * cubegrid_edge_unitlength * cubegrid_edge_unitlength;
		const float system_volume = deltax * deltay * deltaz;
		const int unoccupied_boxes = total_volume_elements - boxes_occupied_by_perc_cluster;

		perc_vol_frac = (single_particle_volume * num_perc_particles) / (single_gridcube_volume * boxes_occupied_by_perc_cluster);
		non_perc_vol_frac = (single_particle_volume * non_perc_particles) / (single_gridcube_volume * unoccupied_boxes);
		frac_percolated = (float) boxes_occupied_by_perc_cluster / total_volume_elements;

		logfile << "Volume Fraction and Counts after first pass through large-scale grid: " << endl;
		logfile << "System volume: " << system_volume << endl;
		logfile << "Single Particle Volume: " << single_particle_volume << endl;
		logfile << "Single Gridcube Volume: " << single_gridcube_volume << endl;
		logfile << "Total Gridcubes: " << total_volume_elements << endl;
		logfile << "Occupied Gridcubes: " << boxes_occupied_by_perc_cluster << endl;
		logfile << "Occupied Gridcube Fraction = Fraction of Total Volume in Percolated Cluster: " << frac_percolated << endl;
		logfile << "Internal Volume Fraction of Percolated Cluster: " << perc_vol_frac << endl;
		logfile << "Internal Volume Fraction of Non-percolated Particles: " << non_perc_vol_frac << endl;

		delete [] discrete_positions;
		discrete_positions = NULL;

		delete [] perc_cluster_particles;
		perc_cluster_particles = NULL;
	}

	//17. Housekeeping cleanup of memory

	delete [] particle_list;
	particle_list = NULL;

	delete [] cluster_list;
	cluster_list = NULL;

	return 0;
}

int neighborcount_3D(const bool (*discrete_positions), int i, int j, int k, const int num_elements_on_grid_edge) {
	//bounds check: repeat values along the boundary
	if(i==0) {i=1;}
	if(j==0) {j=1;}
	if(k==0) {k=1;}
	if(i==num_elements_on_grid_edge) {i=num_elements_on_grid_edge-1;}
	if(j==num_elements_on_grid_edge) {j=num_elements_on_grid_edge-1;}
	if(k==num_elements_on_grid_edge) {k=num_elements_on_grid_edge-1;}

	int neighborcount = discrete_positions[Index_3Dcube_to_1D(i-1,j-1,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j-1,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j-1,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i-1,j,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i-1,j+1,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j+1,k-1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j+1,k-1,num_elements_on_grid_edge)] +

						discrete_positions[Index_3Dcube_to_1D(i-1,j-1,k,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j-1,k,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j-1,k,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i-1,j,k,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j,k,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i-1,j+1,k,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j+1,k,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j+1,k,num_elements_on_grid_edge)] +

						discrete_positions[Index_3Dcube_to_1D(i-1,j-1,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j-1,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j-1,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i-1,j,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i-1,j+1,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i,j+1,k+1,num_elements_on_grid_edge)] +
						discrete_positions[Index_3Dcube_to_1D(i+1,j+1,k+1,num_elements_on_grid_edge)];
	return neighborcount;
}

//Modification History
//Version 1.0: calculates 3-D g(r) (started August 23, 2005)
//Version 1.1: Integrated into multiple-stack framework of Struct3Dt
//(C)opyright 2005 Peter J. Lu. All Rights Reserved. Last Modified, August 28, 2005
int gr3_corr_analysis(const int totalnumparticles, float (*positionlist)[3], float *gr3_output, float *corr_output,
					  const int totalstacknumber, const int currentstacknumber,  const float max_radius_gr3, const int num_bins,
					  ofstream &logfile) { //, ofstream &gr3datafile, ofstream &corrdatafile) {

	//Create data structure of particles for analysis
	//column 0: x-position (microns), column 1: y-position (microns), column 2: z-position (microns)
	float (*particle_list)[4] = new float[totalnumparticles][4];
	int j=0;
	for(j=0; j < totalnumparticles; j++) {
		particle_list[j][0] = positionlist[j][0];	//x-coordinate
		particle_list[j][1] = positionlist[j][1];	//y-coordinate
		particle_list[j][2] = positionlist[j][2];	//z-coordinate
		particle_list[j][3] = 0;	//distance to nearest boundary plane
	}


	//calculate maximum values for extents (system size) in x, y and z
	float minx = 1000, maxx = 0, miny = 1000, maxy = 0, minz = 1000, maxz = 0;
	for (j=0; j < totalnumparticles; j++) {
		if(particle_list[j][0] < minx) {
			minx = particle_list[j][0];
		}
		if(particle_list[j][1] < miny) {
			miny = particle_list[j][1];
		}
		if(particle_list[j][2] < minz) {
			minz = particle_list[j][2];
		}
		if(particle_list[j][0] > maxx) {
			maxx = particle_list[j][0];
		}
		if(particle_list[j][1] > maxy) {
			maxy = particle_list[j][1];
		}
		if(particle_list[j][2] > maxz) {
			maxz = particle_list[j][2];
		}
	}
	const float deltax = maxx - minx;
	const float deltay = maxy - miny;
	const float deltaz = maxz - minz;

	logfile << "Extents in x,y,z: " << deltax << ", " << deltay << ", " << deltaz << endl;
	const float volume = deltax * deltay * deltaz;
	const float density = totalnumparticles / volume;
	float minlength = 0;
	if(deltax < deltay && deltax < deltaz) {
		minlength = deltax;
	}
	else if(deltay < deltaz && deltay < deltax) {
		minlength = deltay;
	}
	else {
		minlength = deltaz;
	}

	//calculate bin widths and maximum radius for correlation function
	const float max_radius_corr = minlength / 2;
	const float max_radius_corr2 = max_radius_corr * max_radius_corr;
	const float binwidth_corr = max_radius_corr / num_bins;
	logfile << "c(r) calculated out to r = " << max_radius_corr<< " in " << num_bins << " bins;";
	logfile << " each bin is " << binwidth_corr << " wide." << endl;

	//calculate bin widths and maximum radius for g(r) function
	const float binwidth_gr3 = max_radius_gr3 / num_bins;
	const float max_radius_gr3_2 = max_radius_gr3 * max_radius_gr3;
	logfile << "g(r) calculated out to r = " << max_radius_gr3 << " in " << num_bins << " bins;" ;
	logfile << " each bin is " << binwidth_gr3 << " wide." << endl;

	//Calculate distance for each point from the nearest boundary, for correlation function calculation
	float xminbound=0, yminbound=0, zminbound=0;	//distance to nearest boundary in x, y and z.
	float xpos=0, ypos=0, zpos=0;
	for(j = 0; j < totalnumparticles; j++) {
		xpos = particle_list[j][0];
		ypos = particle_list[j][1];
		zpos = particle_list[j][2];
		xminbound = smaller(xpos-minx,maxx-xpos);
		yminbound = smaller(ypos-miny,maxy-ypos);
		zminbound = smaller(zpos-minz,maxz-zpos);
		particle_list[j][3] = smallestf(xminbound, yminbound, zminbound);
	}

	//Create array to hold number of interparticle separation distances as a function of r^2
	//Do not use sqrt until the end to save computation: instead,  have num_bins^2 number of bins
	const int numbins2 = num_bins * num_bins;

	//create and initialize r^2-data arrays, where index corresponds to r^2 values
	//note that correlation and g(r) are calculated over separate ranges
	//but we combine the operations where we can, to increase performance
	float (*g_r2)[2] = new float[numbins2][2];
	int *count_r2 = new int[numbins2];			//stores total particle count for i-j bond lengths^2
	int k=0;
	for(k=0; k<numbins2; k++) {
		g_r2[k][0]=0;
		g_r2[k][1]=0;
		count_r2[k]=0;
	}

	//create and initialize r-data arrays, where index corresponds to r values
	float (*gr3)[2] = new float[num_bins][2];		//holds radial distribution function g(r)
	int *count_r = new int[num_bins];				//holds counts(r)
	float (*cr)[3] = new float[num_bins][3];		//holds correlation function c(r)
	for(k=0; k<num_bins; k++) {
		gr3[k][0]=0;
		gr3[k][1]=0;
		count_r[k]=0;
		cr[k][0]=0;
		cr[k][1]=0;
		cr[k][2]=0;			//stores number of particles (outer loop) that have contributed to this bin
	}

	//pass through particle list and calculate distances between all pairs
	int i=0, r2_norm_gr3 = 0, r2_norm_corr = 0, max_rc_norm = 0;
	float r2=0, inv_area=0;
	float dx=0, dy=0, dz=0, max_rc = 0, max_rc2 = 0;
	for(i=0; i<totalnumparticles; i++) {
		//count number of ith particles contributing to the sum of i-j bond lengths (for normalization)
		//note that this count goes only with the radius (not the radius^2)
		max_rc = particle_list[i][3];		//maximum radius for correlation (different from that for g(r) )
											//defined for each particle as the distance to the closest boundary)
		max_rc2 = max_rc * max_rc;		//maximum radius^2 for correlation (different from that for g(r))
		max_rc_norm = num_bins * max_rc / max_radius_corr;
		for(int n=0; n<max_rc_norm; n++) {
			cr[n][2]++;						//increase count by one for each bin; accounts for which particles contribute to averages
		}
		for(j=i+1; j<totalnumparticles; j++) {
			//calculates radial distance between the two particles (bond distance)
			dx = particle_list[i][0]-particle_list[j][0];
			dy = particle_list[i][1]-particle_list[j][1];
			dz = particle_list[i][2]-particle_list[j][2];
			r2 = dx*dx + dy*dy + dz*dz;
			//check to make sure distance is within range that we're interested in calculating
			//perform sum for g(r)
			if(r2 < max_radius_gr3_2) {
				//Normalize actual r^2 value into the range [0 max_radius_gr3] in units of num_bins^2.
				//Then round (note that r2_norm is an INTEGER): avoid passing through entire r^2 array (direct access instead)
				r2_norm_gr3 = numbins2 * r2 / max_radius_gr3_2;
				//calculate "inverse area"
				inv_area = 1 / ( (deltax-fabs(dx)) * (deltay-fabs(dy)) * (deltaz-fabs(dz)) );
				g_r2[r2_norm_gr3][0]+=inv_area;
				g_r2[r2_norm_gr3][1]+=inv_area * inv_area;
			}
			//perform sum for correlation
			if(r2 < max_rc2) {
				r2_norm_corr = numbins2 * r2 / max_radius_corr2;
				count_r2[r2_norm_corr]++;
			}
		}
	}

	//do summation of the r^2 bins and combine, to get sum over r bins
	gr3[0][0]=g_r2[0][0];
	gr3[0][1]=g_r2[0][1];
	count_r[0]=count_r2[0];
	int p=0, q=0;
	for(p=1; p<num_bins; p++) {
		for(q=p*p; q<(p+1)*(p+1); q++) {
			gr3[p][0]+=g_r2[q][0];
			gr3[p][1]+=g_r2[q][1];
			count_r[p]+=count_r2[q];		//mass within shell at radius r
		}
		//divide counts by number of ith particles contributing
		cr[p][0]=count_r[p]/cr[p][2];
		for(q=0;q<=p;q++) {
			cr[p][1]+=cr[q][0];	//total mass integrated from r=0 to current radius
		}
	}

	//start from the end of the list and check for zero particles contributing
	int zero_row_index = 0;
	for(p=num_bins; p>0; p--) {
		if((int) cr[p][2]==0) {
			zero_row_index = p;
		}
	}

	//write output file
	const float g_scalefactor = 2 * 3.14159265 * density * density * binwidth_gr3;
	float actual_r_gr3 = 0, actual_r_corr;
	for(k=1; k<num_bins; k++) {
		actual_r_gr3 = k * binwidth_gr3+ 0.5 * binwidth_gr3;
		//output gr3 data: radius; g(r); error (Poisson)
		//gr3datafile << currentstacknumber << "\t" << actual_r_gr3 << "\t" << gr3[k][0]/(g_scalefactor * actual_r_gr3 * actual_r_gr3);
		//gr3datafile << "\t" << sqrt(gr3[k][1])/(g_scalefactor * actual_r_gr3 * actual_r_gr3 * sqrt(2)) << "\t" << endl;
		gr3_output[Index_1D(k,0,totalstacknumber+1)] = actual_r_gr3;
		gr3_output[Index_1D(k,currentstacknumber,totalstacknumber+1)] = gr3[k][0]/(g_scalefactor * actual_r_gr3 * actual_r_gr3);

		//output correlation data: counts at r, integrated counts to r, normalized correlation functions, particles contributing
		if(k<zero_row_index) {
			actual_r_corr = k * binwidth_corr + 0.5 * binwidth_corr ;
			//corrdatafile << currentstacknumber << "\t" << actual_r_corr << "\t";
			//corrdatafile << count_r[k][0] << "\t" << count_r[k][1] << "\t";
			//corrdatafile << cr[k][0] << "\t" << cr[k][1] << "\t" << cr[k][2] << endl;
			corr_output[Index_1D(k,0,totalstacknumber+1)] = actual_r_corr;
			corr_output[Index_1D(k,currentstacknumber,totalstacknumber+1)] = cr[k][1];
		}
	}

	//Housekeeping cleanup of memory
	delete [] particle_list;
	particle_list = NULL;
	delete [] gr3;
	gr3 = NULL;
	delete [] g_r2;
	g_r2 = NULL;
	delete [] count_r2;
	count_r2 = NULL;
	delete [] count_r;
	count_r = NULL;
	delete [] cr;
	cr = NULL;

	return 0;
}

int Index_3Dcube_to_1D(const int x, const int y, const int z, const int cubeedgelength) {
	return x + (y * cubeedgelength) + (z * cubeedgelength * cubeedgelength);
}


//takes the origin and size of a gridcube, determines what particles contribute to the volume inside,
//tests to see how much space is inaccessible to a test particle of the same size as the regular particles
float cubegrid_occupied_volume(const float (*positionlist)[3], const int totalnumparticles, const float x_origin, const float y_origin, const float z_origin,
//float cubegrid_occupied_volume(float positionlist[MAXROWS][3], const int totalnumparticles, const float x_origin, const float y_origin, const float z_origin,
							   const float cube_edge_length, const float particle_radius, const float bond_distance) {

	//want to make sure that all particles whose solid volumes fill subgrid boxes of the final gridcube are accounted for,
	//even if their centers are not in the gridcube proper. Expanded search around is necessary,
	//even though only subgrid cubes that are part of the actual gridcube will be counted in the final step

	//first, determine which particles have centers that are either in the gridcube itself, or within one bond length of its face/edge
	//(square check; not formal geometric calculation, since extra particles will just be added to a list and this doesn't hurt anything

	//copy the center coordinates of these particles to a dense list
	//create array of large enough size to handle the maximum possible number of particles with partial volume inside the cube

	//establish size variables
//	const float subgrid_edge_length = 0.25;	//distance in microns for SMALLER subgrid---results should not vary with cube_edge_length if this parameter is fixed
	const int subgrid_edge_numcubes = 20;	//number of subgrid units in each cubegrid (play around with this)
	const float subgrid_edge_length = (float) cube_edge_length / subgrid_edge_numcubes;	//actual length, in microns, of the subgrid cube edge length


	//all discrete lengths are given in units of the subgrid edge length, as constant integers
	const int particle_radius_discrete = (int) ceil(particle_radius / subgrid_edge_length);
	const int half_bond_distance_discrete = (int) ceil(0.5 * bond_distance / subgrid_edge_length);			//note that this is a little more than the radius
//	const int subgrid_edge_numcubes = (int) cube_edge_length / subgrid_edge_length;				//for final subgrid, where all elements are within the gridcube
	const int expanded_subgrid_edge_numcubes = subgrid_edge_numcubes + 2 * half_bond_distance_discrete;		//for expanded subgrid, to check for particles whose centers are
																									//outside the final subgrid, but still contribute to filling boxes
	const int final_subgrid_total_cubes = subgrid_edge_numcubes * subgrid_edge_numcubes * subgrid_edge_numcubes;
	const int expanded_subgrid_total_cubes = expanded_subgrid_edge_numcubes * expanded_subgrid_edge_numcubes * expanded_subgrid_edge_numcubes;

	//check for particles whose centers fall into the expanded subgrid
	const int max_particles_in_gridcube = (int) 20 * (expanded_subgrid_edge_numcubes * expanded_subgrid_edge_numcubes * expanded_subgrid_edge_numcubes)
		/ (particle_radius_discrete * particle_radius_discrete * particle_radius_discrete);
	int particles_in_gridcube=0;
	float (*intersecting_particle_centers)[3] = new float[max_particles_in_gridcube][3];
	int i=0;
	for(i=0; i<totalnumparticles; i++) {
		//bounds check to see if particle is in gridcube or any of the surrounding gridcubes; if so, transfer the appropriate coordinates
		if( (positionlist[i][0] > x_origin - bond_distance) && (positionlist[i][0] < x_origin + cube_edge_length + bond_distance) &&
			(positionlist[i][1] > y_origin - bond_distance) && (positionlist[i][1] < y_origin + cube_edge_length + bond_distance) &&
			(positionlist[i][2] > z_origin - bond_distance) && (positionlist[i][2] < z_origin + cube_edge_length + bond_distance) ) {
			if(particles_in_gridcube < max_particles_in_gridcube) {
				intersecting_particle_centers[particles_in_gridcube][0] = positionlist[i][0];
				intersecting_particle_centers[particles_in_gridcube][1] = positionlist[i][1];
				intersecting_particle_centers[particles_in_gridcube][2] = positionlist[i][2];
				particles_in_gridcube++;
			}
			else {
				cout << "Error: Too many particles in gridcube!" << endl;
			}
		}
	}

	//create subgrid cubic array, which contains boolean true if the subcube is in the free volume; more than a particle radius from all particles,
	//alternatively, if a test particle centered at this box does not overlap other particles (only particle center considered): for the EXPANDED grid
	bool *subcube_is_in_freevolume = new bool[expanded_subgrid_total_cubes];
	for(i=0; i<expanded_subgrid_total_cubes; i++) {
		subcube_is_in_freevolume[i]= false;		//initialize to initial state of false; that is, the subcube is assumed not to be part of the inside of the cluster
	}

	//First pass: assign boxes to free volume if a particle in this box does not touch any particles in the percolated cluster
	float x_pos =0, y_pos =0, z_pos =0, r2=0;
	int j=0, k=0, p=0;
	bool subcube_is_near_particle = false;
	const float threshold_r2 = 4 * particle_radius * particle_radius;	//square the particle diameter so we don't have to do a square root within the innermost loop
	for(i=0; i<expanded_subgrid_edge_numcubes; i++) {
		for(j=0; j<expanded_subgrid_edge_numcubes; j++) {
			for(k=0; k<expanded_subgrid_edge_numcubes; k++) {
				subcube_is_near_particle = false;

				//establish position of test particle; offset by physical origin, and difference between final and expanded subgrids
				x_pos = x_origin + (i - half_bond_distance_discrete) * subgrid_edge_length;
				y_pos = y_origin + (j - half_bond_distance_discrete) * subgrid_edge_length;
				z_pos = z_origin + (k - half_bond_distance_discrete) * subgrid_edge_length;

				//run through list of possible particles, and determine if there are any intersections with those
				for(p=0; p<particles_in_gridcube; p++) {
					r2 = (x_pos - intersecting_particle_centers[p][0]) * (x_pos - intersecting_particle_centers[p][0]) +
						 (y_pos - intersecting_particle_centers[p][1]) * (y_pos - intersecting_particle_centers[p][1]) +
						 (z_pos - intersecting_particle_centers[p][2]) * (z_pos - intersecting_particle_centers[p][2]);
					if (r2 < threshold_r2) {
						subcube_is_near_particle = true;
					}
				}

				if (subcube_is_near_particle == false) {
					subcube_is_in_freevolume[Index_3Dcube_to_1D(i,j,k,expanded_subgrid_edge_numcubes)]=true;
				}
			}
		}
	}
	delete [] intersecting_particle_centers;
	intersecting_particle_centers = NULL;

	//take into account full particle volume
	bool *subgrid_freevolume_full = new bool[expanded_subgrid_total_cubes];

	//initialize by copying occupancy values from center-only subgrid
	for(i=0; i<expanded_subgrid_total_cubes; i++) {
		subgrid_freevolume_full[i] = subcube_is_in_freevolume[i];
	}

	//Second pass: for every filled subgrid cube (where a putative test particle center should fit), fill in surrounding subgrid cubes to account for particle volume
	int x=0, y=0, z=0;
	int xmin=0, ymin=0, zmin=0, xmax=0, ymax=0, zmax=0;
	float delta_x = 0, delta_y = 0, delta_z = 0;
	float x_pos_temp = 0, y_pos_temp = 0, z_pos_temp =0;
	for(i=0; i<expanded_subgrid_edge_numcubes; i++) {
		for(j=0; j<expanded_subgrid_edge_numcubes; j++) {
			for(k=0; k<expanded_subgrid_edge_numcubes; k++) {
				//only calculate surrounding subgrid cubes for subgrid cubes where:
				//1. a particle will fit without touching any other particles
				//2. the subgrid is not an "internal" one with 26 neighbors, to reduce duplicate calculation
				if(subcube_is_in_freevolume[Index_3Dcube_to_1D(i,j,k,expanded_subgrid_edge_numcubes)] == true
					&& neighborcount_3D(subcube_is_in_freevolume,i,j,k,expanded_subgrid_edge_numcubes) < 26
					) {

					//do bounds check for inner loop before array, to increase performance
					xmin = i - particle_radius_discrete;
					if(xmin < 0) { xmin = 0; }
					ymin = j - particle_radius_discrete;
					if(ymin < 0) { ymin = 0; }
					zmin = k - particle_radius_discrete;
					if(zmin < 0) { zmin = 0; }

					xmax = i+particle_radius_discrete;
					if(xmax > expanded_subgrid_edge_numcubes) { xmax = expanded_subgrid_edge_numcubes; }
					ymax = j+particle_radius_discrete;
					if(ymax > expanded_subgrid_edge_numcubes) { ymax = expanded_subgrid_edge_numcubes; }
					zmax = k+particle_radius_discrete;
					if(zmax > expanded_subgrid_edge_numcubes) { zmax = expanded_subgrid_edge_numcubes; }

					for(x=xmin; x<xmax; x++) {
						for(y=ymin; y<ymax; y++) {
							for(z=zmin; z<zmax; z++) {
								//determine whether subgrid box is within a particle radius in 3D of a legitimate particle center;
								//old code shown below for clarity, which has been inlined for performance
								//delta_x = (x - i) * subgrid_edge_length; delta_y = (y - j) * subgrid_edge_length; delta_z = (z - k) * subgrid_edge_length;
								//r2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z; if (r2 < particle_radius * particle_radius) {
								if( ((x-i) * (x-i) + (y-j) * (y-j) + (z-k) * (z-k)) * (subgrid_edge_length * subgrid_edge_length) < (particle_radius * particle_radius) ) {
									subgrid_freevolume_full[Index_3Dcube_to_1D(x,y,z,expanded_subgrid_edge_numcubes)]=true;
								}
							}
						}
					}
				}
			}
		}
	}

	delete [] subcube_is_in_freevolume;
	subcube_is_in_freevolume = NULL;

	//copy subset of expanded subgrid array into appropriate final subgrid array
	bool *subgrid_final = new bool[final_subgrid_total_cubes];
	for(i = half_bond_distance_discrete; i<subgrid_edge_numcubes + half_bond_distance_discrete; i++) {
		for(j = half_bond_distance_discrete; j<subgrid_edge_numcubes + half_bond_distance_discrete; j++) {
			for(k = half_bond_distance_discrete; k<subgrid_edge_numcubes + half_bond_distance_discrete; k++) {
				subgrid_final[Index_3Dcube_to_1D(i-half_bond_distance_discrete,j-half_bond_distance_discrete,k-half_bond_distance_discrete,subgrid_edge_numcubes)]
					= subgrid_freevolume_full[Index_3Dcube_to_1D(i,j,k,expanded_subgrid_edge_numcubes)];
			}
		}
	}
	delete [] subgrid_freevolume_full;
	subgrid_freevolume_full = NULL;

	//count up cube grids that are part of the cluster
	int num_subcubes_in_freevolume=0;
	for(i=0; i<final_subgrid_total_cubes; i++) {
		if(subgrid_final[i] == true) {
			num_subcubes_in_freevolume++;
		}
	}
	delete [] subgrid_final;
	subgrid_final = NULL;

	float volume_in_cluster = (final_subgrid_total_cubes - num_subcubes_in_freevolume) * (cube_edge_length * cube_edge_length * cube_edge_length / final_subgrid_total_cubes);
	return volume_in_cluster;
}



//Takes an (x,y,z) pair and identifies it with the location of a box in an integer grid, which has been changed to a 1D list for performance reasons
void discretize_positions(const float (*positionlist)[3], bool *discrete_positions, const int total_particles, const int totalgrid_edgelength, const float edge_unitlength, const float particle_radius) {


	/*
	//OLD version: identifies gridcubes which contain a particle center
	int xcoord = 0, ycoord = 0, zcoord = 0, index_1D = 0;
	for(int i=0; i<total_particles; i++) {
		xcoord = (int) (positionlist[i][0] / edge_unitlength);
		ycoord = (int) (positionlist[i][1] / edge_unitlength);
		zcoord = (int) (positionlist[i][2] / edge_unitlength);
		index_1D = (int) Index_3Dcube_to_1D(xcoord, ycoord, zcoord, totalgrid_edgelength);
		discrete_positions[index_1D]=true;
	}
	*/

	//NEW version: For each particle, finds the grid cube at its center. Then checks surrounding gridcubes to see if the particle volume fills it
	int discrete_radius = (int) 1 + ceil(particle_radius / edge_unitlength);		//radius in units of edge_unitlength for addressing in grid-cube space

	for(int l=0; l<total_particles; l++) {
		//read particle coordinates
		float particle_x_pos = positionlist[l][0];
		float particle_y_pos = positionlist[l][1];
		float particle_z_pos = positionlist[l][2];

		//identify gridcube that contains particle center
		int x_gridpos = (int) floor(particle_x_pos / edge_unitlength);
		int y_gridpos = (int) floor(particle_y_pos / edge_unitlength);
		int z_gridpos = (int) floor(particle_z_pos / edge_unitlength);

		bool status_boxfilled = false;
		//coordinates for bounds of the given cube
		float box_x_min = 0;
		float box_y_min = 0;
		float box_z_min = 0;
		float box_x_max = 0;
		float box_y_max = 0;
		float box_z_max = 0;

		//now, check surrounding boxes:
		for(int i = x_gridpos - discrete_radius; i<= x_gridpos + discrete_radius; i++) {
			//bounds check to see whether this box is actually within the grid
			if(i >=0 && i < totalgrid_edgelength) {
				for(int j = y_gridpos - discrete_radius; j<= y_gridpos + discrete_radius; j++) {
					if(j >=0 && j < totalgrid_edgelength) {
						for(int k = z_gridpos - discrete_radius; k <= z_gridpos + discrete_radius; k++) {
							if(k >=0 && k < totalgrid_edgelength) {
								//calculate 1-D index
								int index_1D = Index_3Dcube_to_1D(i, j, k, totalgrid_edgelength);

								//first check to see if box is already filled; if so, no need for more analysis
								if(discrete_positions[index_1D] == false) {

									//check to see if any part of the particle is contained in the box
									bool status_boxfilled = false;

									box_x_min = i * edge_unitlength;
									box_y_min = j * edge_unitlength;
									box_z_min = k * edge_unitlength;
									box_x_max = (i + 1) * edge_unitlength;
									box_y_max = (j + 1) * edge_unitlength;
									box_z_max = (k + 1) * edge_unitlength;

									//assume that box is cornered at (i * edge_unitedgelength, j * edge_unitedgelength, k * edge_unitlength)
									//check faces
									if( (abs(particle_x_pos - box_x_min) < particle_radius || abs(particle_x_pos - box_x_max) < particle_radius)
										&& (particle_y_pos > box_y_min) && (particle_y_pos < box_y_max)
										&& (particle_z_pos > box_z_min) && (particle_z_pos < box_z_max) ) {
										status_boxfilled = true;
									}

									if( (abs(particle_y_pos - box_y_min) < particle_radius || abs(particle_y_pos - box_y_max) < particle_radius)
										&& (particle_x_pos > box_x_min) && (particle_x_pos < box_x_max)
										&& (particle_z_pos > box_z_min) && (particle_z_pos < box_z_max) ) {
										status_boxfilled = true;
									}

									if( (abs(particle_z_pos - box_z_min) < particle_radius || abs(particle_z_pos - box_z_max) < particle_radius)
										&& (particle_x_pos > box_x_min) && (particle_x_pos < box_x_max)
										&& (particle_y_pos > box_y_min) && (particle_y_pos < box_y_max) ) {
										status_boxfilled = true;
									}

									//check edges
									if( (particle_z_pos > box_z_min) && (particle_z_pos < box_z_max) &&
										( (delta_r_2D(particle_x_pos, box_x_min, particle_y_pos, box_y_min) < particle_radius) ||
										(delta_r_2D(particle_x_pos, box_x_max, particle_y_pos, box_y_min) < particle_radius) ||
										(delta_r_2D(particle_x_pos, box_x_min, particle_y_pos, box_y_max) < particle_radius) ||
										(delta_r_2D(particle_x_pos, box_x_max, particle_y_pos, box_y_max) < particle_radius) ) ) {
										status_boxfilled = true;
									}

									if( (particle_y_pos > box_y_min) && (particle_y_pos < box_y_max) &&
										( (delta_r_2D(particle_x_pos, box_x_min, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_2D(particle_x_pos, box_x_max, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_2D(particle_x_pos, box_x_min, particle_z_pos, box_z_max) < particle_radius) ||
										(delta_r_2D(particle_x_pos, box_x_max, particle_z_pos, box_z_max) < particle_radius) ) ) {
										status_boxfilled = true;
									}

									if( (particle_x_pos > box_x_min) && (particle_x_pos < box_x_max) &&
										( (delta_r_2D(particle_y_pos, box_y_min, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_2D(particle_y_pos, box_y_max, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_2D(particle_y_pos, box_y_min, particle_z_pos, box_z_max) < particle_radius) ||
										(delta_r_2D(particle_y_pos, box_y_max, particle_z_pos, box_z_max) < particle_radius) ) ) {
										status_boxfilled = true;
									}
									//check corners
									if( (delta_r_3D(particle_x_pos, box_x_min, particle_y_pos, box_y_min, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_3D(particle_x_pos, box_x_max, particle_y_pos, box_y_min, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_3D(particle_x_pos, box_x_min, particle_y_pos, box_y_max, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_3D(particle_x_pos, box_x_max, particle_y_pos, box_y_max, particle_z_pos, box_z_min) < particle_radius) ||
										(delta_r_3D(particle_x_pos, box_x_min, particle_y_pos, box_y_min, particle_z_pos, box_z_max) < particle_radius) ||
										(delta_r_3D(particle_x_pos, box_x_max, particle_y_pos, box_y_min, particle_z_pos, box_z_max) < particle_radius) ||
										(delta_r_3D(particle_x_pos, box_x_min, particle_y_pos, box_y_max, particle_z_pos, box_z_max) < particle_radius) ||
										(delta_r_3D(particle_x_pos, box_x_max, particle_y_pos, box_y_max, particle_z_pos, box_z_max) < particle_radius) ){
										status_boxfilled = true;
									}

									//assign status
									if(status_boxfilled == true) {
										discrete_positions[index_1D]=status_boxfilled;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

float delta_r_2D(const float x1, const float x2, const float y1, const float y2) {
	float delta_r = 0;
	delta_r = sqrt( ((x1-x2) * (x1-x2)) + ((y1-y2) * (y1-y2)) );
	return delta_r;
}

float delta_r_3D(const float x1, const float x2, const float y1, const float y2, const float z1, const float z2) {
	float delta_r = 0;
	delta_r = sqrt( ((x1-x2) * (x1-x2)) + ((y1-y2) * (y1-y2)) + ((z1-z2) * (z1-z2)) );
	return delta_r;
}


void copy_discrete_positions(bool *discrete_positions_source, bool *discrete_positions_target, const int total_volume_elements) {
	for(int i=0; i < total_volume_elements; i++) {
		discrete_positions_target[i] = discrete_positions_source[i];
	}
}

void clear_discrete_positions(bool *discrete_positions, const int total_volume_elements) {
	for(int i=0; i < total_volume_elements; i++) {
		discrete_positions[i] = false;
	}
}

void total_z_counts(bool *discrete_positions, int *z_counts, const int totalstacknumber, const int currentstacknumber,
					const int totalgrid_edgelength) {
	int zcount = 0;
	for(int z=0; z< totalgrid_edgelength; z++) {
		z_counts[Index_1D(z,0,totalstacknumber+1)] = z;
		zcount = 0;
		for(int x=0; x<totalgrid_edgelength; x++) {
			for(int y=0; y<totalgrid_edgelength; y++) {
				zcount += discrete_positions[Index_3Dcube_to_1D(x,y,z,totalgrid_edgelength)];
			}
		}
		z_counts[Index_1D(z,currentstacknumber,totalstacknumber+1)] = zcount;
	}
}


float cross_correlation(bool *discrete_positions1, bool *discrete_positions2, const int total_volume_elements) {
	//this presumes that both discrete_positions arrays are the same size = total_volume_elements
	float cross_corr = 0, sumtotal=0;
	for(int i=0; i< total_volume_elements; i++) {
		sumtotal+=discrete_positions1[i];
		cross_corr+=discrete_positions1[i] * discrete_positions2[i];
	}
//	cout << sumtotal << "\t" << cross_corr << "\t" << cross_corr/sumtotal << endl;
	return cross_corr / sumtotal;
}

float systemvolume(float (*positionlist)[3], const int total_particles) {
	float minx = 1000, maxx = 0, miny = 1000, maxy = 0, minz = 1000, maxz = 0;
	for (int j=0; j < total_particles; j++) {
		if(positionlist[j][0] < minx) {
			minx = positionlist[j][0];
		}
		if(positionlist[j][1] < miny) {
			miny = positionlist[j][1];
		}
		if(positionlist[j][2] < minz) {
			minz = positionlist[j][2];
		}
		if(positionlist[j][0] > maxx) {
			maxx = positionlist[j][0];
		}
		if(positionlist[j][1] > maxy) {
			maxy = positionlist[j][1];
		}
		if(positionlist[j][2] > maxz) {
			maxz = positionlist[j][2];
		}
	}
	float deltax = maxx - minx;
	float deltay = maxy - miny;
	float deltaz = maxz - minz;
	return deltax * deltay * deltaz;
}

int chains_analysis(const int totalnumparticles, float (*positionlist)[3], float *chains_output,
					const int totalstacknumber, const int currentstacknumber, const int total_number_of_shells,
					const float shell_width, ofstream &logfile) {
	//Create data structure of particles for analysis
	//column 0: x-position (microns), column 1: y-position (microns), column 2: z-position (microns)
	float (*particle_list)[4] = new float[totalnumparticles][4];	//all particles
	float (*particles_in_shell_list)[3] = new float[totalnumparticles][3];	//temporary array for holding data to pass onto spherical-shell cluster counter
	int j=0;
	for(j = 0; j < totalnumparticles; j++) {
		particle_list[j][0] = positionlist[j][0];	//x-coordinate
		particle_list[j][1] = positionlist[j][1];	//y-coordinate
		particle_list[j][2] = positionlist[j][2];	//z-coordinate
		particle_list[j][3] = 0;	//distance to nearest boundary plane
		particles_in_shell_list[j][0] = 0;
		particles_in_shell_list[j][1] = 0;
		particles_in_shell_list[j][2] = 0;
	}

	//calculate maximum values for extents (system size) in x, y and z
	float minx = 1000, maxx = 0, miny = 1000, maxy = 0, minz = 1000, maxz = 0;
	for (j=0; j < totalnumparticles; j++) {
		if(particle_list[j][0] < minx) {
			minx = particle_list[j][0];
		}
		if(particle_list[j][1] < miny) {
			miny = particle_list[j][1];
		}
		if(particle_list[j][2] < minz) {
			minz = particle_list[j][2];
		}
		if(particle_list[j][0] > maxx) {
			maxx = particle_list[j][0];
		}
		if(particle_list[j][1] > maxy) {
			maxy = particle_list[j][1];
		}
		if(particle_list[j][2] > maxz) {
			maxz = particle_list[j][2];
		}
	}
	const float deltax = maxx - minx;
	const float deltay = maxy - miny;
	const float deltaz = maxz - minz;

	const float volume = deltax * deltay * deltaz;
	const float density = totalnumparticles / volume;
	float minlength = 0;
	if(deltax < deltay && deltax < deltaz) {
		minlength = deltax;
	}
	else if(deltay < deltaz && deltay < deltax) {
		minlength = deltay;
	}
	else {
		minlength = deltaz;
	}

	//Calculate distance for each point from the nearest boundary, for correlation function calculation
	float xminbound=0, yminbound=0, zminbound=0;	//distance to nearest boundary in x, y and z.
	float xpos=0, ypos=0, zpos=0;
	for(j = 0; j < totalnumparticles; j++) {
		xpos = particle_list[j][0];
		ypos = particle_list[j][1];
		zpos = particle_list[j][2];
		xminbound = smaller(xpos-minx,maxx-xpos);
		yminbound = smaller(ypos-miny,maxy-ypos);
		zminbound = smaller(zpos-minz,maxz-zpos);
		particle_list[j][3] = smallestf(xminbound, yminbound, zminbound);
	}

	//create arrays to hold number of total number of clusters in a given spherical shell, and
	//the number of particles contributing to that cluster count (to normalize for different distances from boundaries)
	int num_shell_bins = 1.5 * minlength / shell_width;
	int *particles_contributing_to_shell = new int[num_shell_bins];
	int *total_clusters_in_shell = new int[num_shell_bins];

	int s=0;
	for(s=0; s<num_shell_bins; s++) {
		particles_contributing_to_shell[s] = 0;
		total_clusters_in_shell[s] = 0;
	}

	//pass through particle list and calculate distances between all pairs
	float r2=0;
	float dx=0, dy=0, dz=0;


	for(int i=0; i<totalnumparticles; i++) {
		//first loop passes through particle array to determine r2 with other particles
		//for each particle, create array with distances to other particles, where row number is the particle identifier
		float *distance_list = new float[totalnumparticles];
		for(j=0; j<totalnumparticles; j++) {
			//re-clear particles_in_shell_list
			particles_in_shell_list[j][0] = 0;
			particles_in_shell_list[j][1] = 0;
			particles_in_shell_list[j][2] = 0;

			//calculates radial distance between the two particles (bond distance)
			dx = particle_list[i][0]-particle_list[j][0];
			dy = particle_list[i][1]-particle_list[j][1];
			dz = particle_list[i][2]-particle_list[j][2];
			r2 = dx*dx + dy*dy + dz*dz;
			distance_list[j] = sqrt(r2);
		}

		//second loop goes over radial distance from that particle
		//for each shell, calculate number of clusters (i.e. chains)

		//calculate number of multiples of shell thickness by dividing distance to nearest boundary by shell_width
		int max_number_of_shells = 2 * (particle_list[i][3] / shell_width) + 1;
		for(s = 0; s < max_number_of_shells; s++) {
			particles_contributing_to_shell[s]++;		//adds contribution for ith particle

			int particles_in_shell_count = 0;	//number of particles in relevant spherical shell
			//go through list and count number of particles in spherical shell between r and r + 2dr (i.e. centered around r + dr)
			for(j=0; j<totalnumparticles; j++) {
				//check to see if that particle is in the shell specified
				float r_min = s * shell_width / 2;
				if( (distance_list[j] >= r_min) && (distance_list[j] < r_min + shell_width)) {
					particles_in_shell_list[particles_in_shell_count][0] = particle_list[j][0];
					particles_in_shell_list[particles_in_shell_count][1] = particle_list[j][1];
					particles_in_shell_list[particles_in_shell_count][2] = particle_list[j][2];
					//increase particle count
					particles_in_shell_count++;
				}
			}

			//call function to return number of clusters in this shell
			int num_clusters = num_clusters_in_shell(particles_in_shell_count, particles_in_shell_list, 4 * shell_width * shell_width);
			total_clusters_in_shell[s] += num_clusters;

//			if(i % 500 == 0) {
//				cout << s << "\t" << num_clusters << endl;
//			}

		}
		delete [] distance_list;
		distance_list = NULL;
	}

	//normalize cluster count by number of particles contributing
	float *normalized_cluster_count = new float[num_shell_bins];
	for(s=0;s<num_shell_bins;s++) {
		if(particles_contributing_to_shell[s]>0) {
			normalized_cluster_count[s] = (float) total_clusters_in_shell[s] / particles_contributing_to_shell[s];
		}
		else {
			normalized_cluster_count[s] =0;
		}
	}

	//copy chains data to output array
	for(s=0;s<num_shell_bins;s++) {
		chains_output[Index_1D(s,0,totalstacknumber+1)] = ((float) s+0.5) * shell_width / 2;	//centered r-coordinate (shell radius)
		chains_output[Index_1D(s,currentstacknumber,totalstacknumber+1)]  = normalized_cluster_count[s];
	}

	//Housekeeping cleanup of memory
	delete [] particle_list;
	particle_list = NULL;
	delete [] particles_in_shell_list;
	particles_in_shell_list = NULL;
	delete [] particles_contributing_to_shell;
	particles_contributing_to_shell = NULL;
	delete [] total_clusters_in_shell;
	total_clusters_in_shell = NULL;
	delete [] normalized_cluster_count;
	normalized_cluster_count = NULL;

	return 0;
}

int num_clusters_in_shell(const int totalnumparticles, float (*positionlist)[3], const float bond_r2) {
	//create new xyz array with just the position information of the particles in the shell of concern

	//column 0: x-position (microns)
	//column 1: y-position (microns)
	//column 2: z-position (microns)
	//column 3:	cluster label (raw)
	float (*particle_list)[4] = new float[totalnumparticles][4];

	//create array to hold index values, which will then be sorted to backpropagate to particle list
	int *cluster_index_list = new int[totalnumparticles];

	//3. Copy particle data, assign cluster number
	int j=0;
	for(j = 0; j < totalnumparticles; j++) {
		particle_list[j][0] = positionlist[j][0];	//copy x-position data
		particle_list[j][1] = positionlist[j][1];	//copy y-position data
		particle_list[j][2] = positionlist[j][2];	//copy z-position data
		particle_list[j][3] = 0;					//cluster label
		cluster_index_list[j] = j;
	}

	//4. Loop through particle list to determine bonding, count nearest neighbors
	int i=0, k=0, oldsmallerindex = 0, oldsmallerindex2 = 0, newsmallerindex=0;
	float r2 = 0;
	float delta_x = 0, delta_y = 0, delta_z = 0;

	for(i=0; i<totalnumparticles; i++) {
		for(j=i+1; j<totalnumparticles; j++) {
			//calculates radial distance between the two particles = bond distance
			r2 = (particle_list[i][0]-particle_list[j][0]) * (particle_list[i][0]-particle_list[j][0]) +
				(particle_list[i][1]-particle_list[j][1]) * (particle_list[i][1]-particle_list[j][1]) +
				(particle_list[i][2]-particle_list[j][2]) * (particle_list[i][2]-particle_list[j][2]);

			//check if bond is within threshold
			if(r2 < bond_r2) {
				//determine if this is first bond being made (no reassignment necessary)
				if(cluster_index_list[j] == j) {
					cluster_index_list[j]=cluster_index_list[i];
				}
				//otherwise, need to propagate the labels
				else {
					oldsmallerindex = cluster_index_list[j];
					oldsmallerindex2 = cluster_index_list[i];
					newsmallerindex = smallest(oldsmallerindex,oldsmallerindex2,cluster_index_list[oldsmallerindex]);
					for(k=0; k<totalnumparticles; k++) {
						if(cluster_index_list[k]==oldsmallerindex || cluster_index_list[k]==oldsmallerindex2) {
							cluster_index_list[k]=newsmallerindex;
						}
					}
				}
			}
		}
	}
	//Assign cluster labels to particle data
	for(k=0; k<totalnumparticles; k++) {
		particle_list[k][3] = cluster_index_list[k];
	}
	delete [] cluster_index_list;
	cluster_index_list = NULL;

	//6. Create new cluster array, sorting size and percolation information
	//column 0: number of particles in that cluster/particle count (i.e. size)
	float *cluster_list = new float[totalnumparticles+1];

	//this array mostly empty array because the cluster numbers are not sorted.
	//row number is the cluster label

	//fill with zeros (minimum cluster label should be zero, from above)
	for(i=0; i<=totalnumparticles; i++) {
			cluster_list[i] =0;
	}

	//pass through particle list to count particles in each cluster
	int cluster_label = 0;
	for(i=0;i<totalnumparticles;i++) {
		//increment particle count
		cluster_label = (int) particle_list[i][3];
		cluster_list[cluster_label]++;
	}
	delete [] particle_list;
	particle_list = NULL;

	//Count number of clusters
	int count = 0;
	for(i = 0; i<=totalnumparticles; i++) {
		if( (int) cluster_list[i] > 2) {	//cluster must have more than two particles (eliminates spurious effects due to monomers/dimers)
			count++;
		}
	}
	delete [] cluster_list;
	cluster_list = NULL;

//	cout << count << " ";

	return count;

}
