
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip> // for std::setw
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {

	double dtime1, speedUp, Eff, cost;
	dtime1 = 0.0000015; //  1.50001e-06 seconds serial execution time

	MPI_Init(&argc, &argv);

	int num_procs;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (num_procs != 4) {
		if (rank == 0) {
			printf("This program requires exactly 4 MPI processes.\n");
		}
		MPI_Finalize();
		return -1;
	}
	int inputImage[6][6] = {
		{1, 2, 3, 4, 5, 6},
		{7, 8, 9, 10, 11, 12},
		{13, 14, 15, 16, 17, 18},
		{19, 20, 21, 22, 23, 24},
		{25, 26, 27, 28, 29, 30},
		{31, 32, 33, 34, 35, 36}
	};

	int kernel[3][3] = {
		{3, 4, 5},
		{6, 7, 8},
		{9, 10, 11}
	};

	int outputImage[6][6] = { 0 };


	int local_outputImage[6] = { 0 };

	auto start = chrono::high_resolution_clock::now();

	//scatter
	int i = 1 + rank;
	for (int j = 1; j < 5; j++) {
		int R1 = 0;
		for (int k = -1; k < 2; k++) {
			for (int m = -1; m < 2; m++) {
				R1 += inputImage[i + k][j + m] * kernel[k + 1][m + 1];
			}
		}
		local_outputImage[j] = R1;
	}
	//gather
	MPI_Gather(local_outputImage, 6, MPI_INT,
		&outputImage[rank+1], 6, MPI_INT,
		0, MPI_COMM_WORLD);


	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = end - start;

	if (rank == 0) {
		cout << "\nSerial execution time: " << dtime1 << " seconds" << endl;
		cout << "Execution time of MPI code: " << elapsed.count() << " seconds" << endl;


		//display output
		cout << "\nOutput Image:" << endl;
		for (int i = 0; i < 6; i++) {
			printf("\n\t| ");
			for (int j = 0; j < 6; j++) {
				cout << setw(5) << outputImage[i][j] << " ";
				printf("|");
			}
			cout << endl;
		}

	
		speedUp = dtime1 / elapsed.count();
		printf("\nSpeedUp = %f\n", speedUp);
		Eff = speedUp / num_procs;
		printf("Efficiency = %f\n", Eff);
		cost = elapsed.count() * num_procs;
		printf("cost = %f\n", cost);

	}

	MPI_Finalize();


	return 0;
}


