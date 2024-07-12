
// ============================ Lab 10 =================================

// Task 1
#include <iostream>
#include <vector>
#include <mpi.h>

#define SIZE 4 /* Size of matrices */

int A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
double start_time, end_time;

void fill_matrix(int m[SIZE][SIZE])
{
	static int n = 0;
	int i, j;
	for (i = 0; i < SIZE; i++)
		for (j = 0; j < SIZE; j++)
			m[i][j] = n++;
}
void print_matrix(int m[SIZE][SIZE])
{
	int i, j = 0;
	for (i = 0; i < SIZE; i++) {
		printf("\n\t| ");
		for (j = 0; j < SIZE; j++)
			printf("%2d ", m[i][j]);
		printf("|");
	}
}
int main(int argc, char* argv[])
{
	int myrank, P, from, to, i, j, k;
	int tag = 666; /* any value will do */
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); /* who am i */
	MPI_Comm_size(MPI_COMM_WORLD, &P); /* number of processors */
	/* Just to use the simple variants of MPI_Gather and MPI_Scatter we */
	/* impose that SIZE is divisible by P. By using the vector versions, */
	/* (MPI_Gatherv and MPI_Scatterv) it is easy to drop this restriction.
	*/

	if (SIZE % P != 0) {
		if (myrank == 0) printf("Matrix size not divisible by number of processors\n");
		MPI_Finalize();
		exit(-1);
	}
	from = myrank * SIZE / P;
	to = (myrank + 1) * SIZE / P;
	/* Process 0 fills the input matrices and broadcasts them to the rest
	*/
	/* (actually, only the relevant stripe of A is sent to each process) */
	if (myrank == 0) {
		fill_matrix(A);
		fill_matrix(B);
	}
	start_time = MPI_Wtime();
	MPI_Bcast(B, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(A, SIZE * SIZE / P, MPI_INT, A[from], SIZE * SIZE / P, MPI_INT, 0,
		MPI_COMM_WORLD);
	printf("computing slice %d (from row %d to %d)\n", myrank, from, to - 1);
	for (i = from; i < to; i++)
		for (j = 0; j < SIZE; j++) {
			C[i][j] = 0;
			for (k = 0; k < SIZE; k++)
				C[i][j] += A[i][k] * B[k][j];
		}
	MPI_Gather(C[from], SIZE * SIZE / P, MPI_INT, C, SIZE * SIZE / P, MPI_INT, 0,
		MPI_COMM_WORLD);
	end_time = MPI_Wtime();
	if (myrank == 0) {
		printf("\n\n");
		print_matrix(A);
		printf("\n\n\t * \n");
		print_matrix(B);
		printf("\n\n\t = \n");
		print_matrix(C);
		printf("\n\n");
		printf("Execution time is: %f\n", end_time - start_time);
	}
	MPI_Finalize();
}






// Ex 2

#include <iostream>
#include <vector>
#include <mpi.h>

using namespace std;

// Function to initialize vectors with random values (for demonstration)
void initializeVectors(vector<int>& v1, vector<int>& v2, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        v1[i] = rand() % 100; // random values between 0 and 99
        v2[i] = rand() % 100;
    }
}

int main(int argc, char** argv) {
    const int vector_size = 512;

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Calculate local size for each process
    int local_size = vector_size / num_procs;
    vector<int> local_v1(local_size);
    vector<int> local_v2(local_size);
    vector<int> local_result(local_size);

    initializeVectors(local_v1, local_v2, local_size);

    for (int i = 0; i < local_size; ++i) {
        local_result[i] = local_v1[i] + local_v2[i];
    }
    if (rank == 0) {
        vector<int> final_result(vector_size);
        for (int i = 0; i < local_size; ++i) {
            final_result[i] = local_result[i];
        }
        for (int p = 1; p < num_procs; ++p) {
            MPI_Recv(&final_result[p * local_size], local_size, 
                MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        cout << "Final result for vector size " << vector_size << " :" << endl;
        for (int i = 0; i < vector_size; ++i) {
            cout << final_result[i] << " ";
        }
        cout << endl;
    }
    else {
        // Send local results to process 0
        MPI_Send(&local_result[0], local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}








// Ex 3

#include <iostream>
#include <vector>
#include <mpi.h>

using namespace std;

// Function to initialize matrix and vector with random values (for demonstration)
void initializeMatrixVector(vector<vector<int>>& A, vector<int>& B, int N) {
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        B[i] = rand() % 100; // random values between 0 and 99
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 100; // random values between 0 and 99
        }
    }
}

int main(int argc, char** argv) {
    const int N = 512; // Assuming a square matrix of size N x N

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Calculate local size for each process
    int local_rows = N / num_procs;

    // Allocate memory for the local parts of matrix A and vector B
    vector<vector<int>> local_A(local_rows, vector<int>(N));
    vector<int> B(N), local_C(local_rows, 0);

    // Process 0 initializes the full matrix A and vector B and distributes them
    if (rank == 0) {
        vector<vector<int>> A(N, vector<int>(N));
        initializeMatrixVector(A, B, N);

        // Distribute rows of A to other processes
        for (int i = 1; i < num_procs; ++i) {
            MPI_Send(&A[i * local_rows][0], local_rows * N, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        // Copy rows for process 0
        for (int i = 0; i < local_rows; ++i) {
            local_A[i] = A[i];
        }
    }
    else {
        // Receive rows of A
        MPI_Recv(&local_A[0][0], local_rows * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Broadcast vector B to all processes
    MPI_Bcast(&B[0], N, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local matrix-vector multiplication
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            local_C[i] += local_A[i][j] * B[j];
        }
    }

    // Gather the partial results from all processes to process 0
    if (rank == 0) {
        vector<int> C(N);
        // Copy local results of process 0
        for (int i = 0; i < local_rows; ++i) {
            C[i] = local_C[i];
        }
        // Receive results from other processes
        for (int i = 1; i < num_procs; ++i) {
            MPI_Recv(&C[i * local_rows], local_rows, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Print the final result vector (for demonstration)
        cout << "Final result vector C:" << endl;
        for (int i = 0; i < N; ++i) {
            cout << C[i] << " ";
        }
        cout << endl;
    }
    else {
        // Send local results to process 0
        MPI_Send(&local_C[0], local_rows, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}





// Ex 5

#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 2048;  // Size of the matrix

void serialTranspose(const vector<vector<int>>& A, vector<vector<int>>& B) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B[j][i] = A[i][j];
}

void parallelTranspose(const vector<vector<int>>& A, vector<vector<int>>& B, int rank, int size) {
    int rows_per_proc = N / size;
    int start = rank * rows_per_proc;
    int end = (rank + 1) * rows_per_proc;

    for (int i = start; i < end; ++i)
        for (int j = 0; j < N; ++j)
            B[j][i] = A[i][j];

    for (int i = 0; i < N; ++i) {
        MPI_Allgather(MPI_IN_PLACE, rows_per_proc, MPI_INT, B[i].data(), rows_per_proc, MPI_INT, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));

    // Initialize matrix A
    if (rank == 0) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                A[i][j] = i * N + j;
    }

    // Broadcast matrix A to all processes
    for (int i = 0; i < N; ++i) {
        MPI_Bcast(A[i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Measure time for serial transpose
    high_resolution_clock::time_point start_serial, end_serial;
    if (rank == 0) {
        start_serial = high_resolution_clock::now();
        serialTranspose(A, B);
        end_serial = high_resolution_clock::now();
    }

    // Measure time for parallel transpose
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start at the same time
    high_resolution_clock::time_point start_parallel = high_resolution_clock::now();
    parallelTranspose(A, B, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish before stopping the clock
    high_resolution_clock::time_point end_parallel = high_resolution_clock::now();

    // Calculate and print execution times and speedup
    if (rank == 0) {
        auto duration_serial = duration_cast<milliseconds>(end_serial - start_serial).count();
        auto duration_parallel = duration_cast<milliseconds>(end_parallel - start_parallel).count();
        double speedup = static_cast<double>(duration_serial) / duration_parallel;

        cout << "For Matrix size = "  << N << endl;
        cout << "Serial execution time: " << duration_serial << " ms" << endl;
        cout << "Parallel execution time: " << duration_parallel << " ms" << endl;
        cout << "Speedup: " << speedup << endl;
    }

    MPI_Finalize();
    return 0;
}






// Ex 4

#include <iostream>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

typedef unsigned long long ull;

// Function to calculate factorial serially
ull serialFactorial(int n) {
    ull result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Function to calculate factorial in parallel using MPI
ull parallelFactorial(int n, int rank, int size) {
    ull local_result = 1;
    for (int i = rank + 1; i <= n; i += size) {
        local_result *= i;
    }

    ull global_result = 1;
    MPI_Reduce(&local_result, &global_result, 1, MPI_UNSIGNED_LONG_LONG, MPI_PROD, 0, MPI_COMM_WORLD);
    return global_result;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 32; // Number to calculate factorial of
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    // Measure time for serial factorial
    high_resolution_clock::time_point start_serial, end_serial;
    ull serial_result;
    if (rank == 0) {
        start_serial = high_resolution_clock::now();
        serial_result = serialFactorial(n);
        end_serial = high_resolution_clock::now();
    }

    // Measure time for parallel factorial
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start at the same time
    high_resolution_clock::time_point start_parallel = high_resolution_clock::now();
    ull parallel_result = parallelFactorial(n, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish before stopping the clock
    high_resolution_clock::time_point end_parallel = high_resolution_clock::now();

    // Calculate and print execution times and speedup
    if (rank == 0) {
        auto duration_serial = duration_cast<milliseconds>(end_serial - start_serial).count();
        auto duration_parallel = duration_cast<milliseconds>(end_parallel - start_parallel).count();
        double speedup = static_cast<double>(duration_serial) / duration_parallel;

        cout << "Serial factorial of " << n << " is: " << serial_result << endl;
        cout << "Parallel factorial of " << n << " is: " << parallel_result << endl;
        cout << "Serial execution time: " << duration_serial << " ms" << endl;
        cout << "Parallel execution time: " << duration_parallel << " ms" << endl;
        cout << "Speedup: " << speedup << endl;
    }

    MPI_Finalize();
    return 0;
}





// Ex 3

#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 128;  // Size of the matrix and vector

void serialMatrixVectorProduct(const vector<vector<int>>& A, const vector<int>& x, vector<int>& y) {
    for (int i = 0; i < N; ++i) {
        y[i] = 0;
        for (int j = 0; j < N; ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
}

void parallelMatrixVectorProduct(const vector<vector<int>>& A, const vector<int>& x, vector<int>& y, int rank, int size) {
    int rows_per_proc = N / size;
    int start = rank * rows_per_proc;
    int end = (rank + 1) * rows_per_proc;

    vector<int> local_y(rows_per_proc, 0);
    for (int i = start; i < end; ++i) {
        local_y[i - start] = 0;
        for (int j = 0; j < N; ++j) {
            local_y[i - start] += A[i][j] * x[j];
        }
    }

    MPI_Gather(local_y.data(), rows_per_proc, MPI_INT, y.data(), rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<int>> A(N, vector<int>(N));
    vector<int> x(N);
    vector<int> y(N);

    // Initialize matrix A and vector x
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = i + j;
            }
            x[i] = i;
        }
    }

    // Broadcast matrix A and vector x to all processes
    for (int i = 0; i < N; ++i) {
        MPI_Bcast(A[i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(x.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // Measure time for serial matrix-vector product
    high_resolution_clock::time_point start_serial, end_serial;
    if (rank == 0) {
        start_serial = high_resolution_clock::now();
        serialMatrixVectorProduct(A, x, y);
        end_serial = high_resolution_clock::now();
    }

    // Measure time for parallel matrix-vector product
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start at the same time
    high_resolution_clock::time_point start_parallel = high_resolution_clock::now();
    parallelMatrixVectorProduct(A, x, y, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish before stopping the clock
    high_resolution_clock::time_point end_parallel = high_resolution_clock::now();

    // Calculate and print execution times and speedup
    if (rank == 0) {
        auto duration_serial = duration_cast<milliseconds>(end_serial - start_serial).count();
        auto duration_parallel = duration_cast<milliseconds>(end_parallel - start_parallel).count();
        double speedup = static_cast<double>(duration_serial) / duration_parallel;

        cout << "For Matrix & Vector size = " << N << endl;
        cout << "Serial execution time: " << duration_serial << " ms" << endl;
        cout << "Parallel execution time: " << duration_parallel << " ms" << endl;
        cout << "Speedup: " << speedup << endl;
    }

    MPI_Finalize();
    return 0;
}




// Ex 2

#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 16384000;  // Size of the vectors

void serialVectorSum(const vector<int>& A, const vector<int>& B, vector<int>& C) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

void parallelVectorSum(const vector<int>& A, const vector<int>& B, vector<int>& C, int rank, int size) {
    int elements_per_proc = N / size;
    int start = rank * elements_per_proc;
    int end = (rank + 1) * elements_per_proc;

    vector<int> local_C(elements_per_proc);
    for (int i = start; i < end; ++i) {
        local_C[i - start] = A[i] + B[i];
    }

    MPI_Gather(local_C.data(), elements_per_proc, MPI_INT, C.data(), elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> A(N);
    vector<int> B(N);
    vector<int> C(N);

    // Initialize vectors A and B
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            A[i] = i;
            B[i] = N - i;
        }
    }

    // Broadcast vectors A and B to all processes
    MPI_Bcast(A.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // Measure time for serial vector sum
    high_resolution_clock::time_point start_serial, end_serial;
    if (rank == 0) {
        start_serial = high_resolution_clock::now();
        serialVectorSum(A, B, C);
        end_serial = high_resolution_clock::now();
    }

    // Measure time for parallel vector sum
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start at the same time
    high_resolution_clock::time_point start_parallel = high_resolution_clock::now();
    parallelVectorSum(A, B, C, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish before stopping the clock
    high_resolution_clock::time_point end_parallel = high_resolution_clock::now();

    // Calculate and print execution times and speedup
    if (rank == 0) {
        auto duration_serial = duration_cast<milliseconds>(end_serial - start_serial).count();
        auto duration_parallel = duration_cast<milliseconds>(end_parallel - start_parallel).count();
        double speedup = static_cast<double>(duration_serial) / duration_parallel;

        cout << "For Vector size = " << N << endl;
        cout << "Serial execution time: " << duration_serial << " ms" << endl;
        cout << "Parallel execution time: " << duration_parallel << " ms" << endl;
        cout << "Speedup: " << speedup << endl;
    }

    MPI_Finalize();
    return 0;
}




// Ex 1

#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 16;  // Size of the matrices

void serialMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallelMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int rank, int size) {
    int rows_per_proc = N / size;
    int start = rank * rows_per_proc;
    int end = (rank + 1) * rows_per_proc;

    vector<vector<int>> local_C(rows_per_proc, vector<int>(N, 0));
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                local_C[i - start][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (int i = 0; i < rows_per_proc; ++i) {
        MPI_Gather(local_C[i].data(), N, MPI_INT, C[start + i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N, 0));

    // Initialize matrices A and B
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = i + j;
                B[i][j] = i - j;
            }
        }
    }

    // Broadcast matrices A and B to all processes
    for (int i = 0; i < N; ++i) {
        MPI_Bcast(A[i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(B[i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Measure time for serial matrix multiplication
    high_resolution_clock::time_point start_serial, end_serial;
    if (rank == 0) {
        start_serial = high_resolution_clock::now();
        serialMatrixMultiplication(A, B, C);
        end_serial = high_resolution_clock::now();
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start at the same time
    high_resolution_clock::time_point start_parallel = high_resolution_clock::now();
    parallelMatrixMultiplication(A, B, C, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish before stopping the clock
    high_resolution_clock::time_point end_parallel = high_resolution_clock::now();

    if (rank == 0) {
        auto duration_serial = duration_cast<milliseconds>(end_serial - start_serial).count();
        auto duration_parallel = duration_cast<milliseconds>(end_parallel - start_parallel).count();
        double speedup = static_cast<double>(duration_serial) / duration_parallel;

        cout << "For Matrix size = " << N << endl;
        cout << "Serial execution time: " << duration_serial << " ms" << endl;
        cout << "Parallel execution time: " << duration_parallel << " ms" << endl;
        cout << "Speedup: " << speedup << endl;
    }

    MPI_Finalize();
    return 0;
}


#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 16;  // Size of the matrices

void serialMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallelMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int rank, int size) {
    int rows_per_proc = N / size;
    int start = rank * rows_per_proc;
    int end = (rank + 1) * rows_per_proc;

    vector<vector<int>> local_C(rows_per_proc, vector<int>(N, 0));
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                local_C[i - start][j] += A[i][k] * B[k][j];
            }
        }
    }

    MPI_Gather(&local_C[0][0], rows_per_proc * N, MPI_INT, &C[0][0], rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N, 0));

    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = i + j;
                B[i][j] = i - j;
            }
        }
    }

    // Broadcast matrices A and B to all processes
    for (int i = 0; i < N; ++i) {
        MPI_Bcast(A[i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(B[i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Measure time for serial matrix multiplication
    high_resolution_clock::time_point start_serial, end_serial;
    if (rank == 0) {
        start_serial = high_resolution_clock::now();
        serialMatrixMultiplication(A, B, C);
        end_serial = high_resolution_clock::now();
    }

    // Measure time for parallel matrix multiplication
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start at the same time
    high_resolution_clock::time_point start_parallel = high_resolution_clock::now();
    parallelMatrixMultiplication(A, B, C, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish before stopping the clock
    high_resolution_clock::time_point end_parallel = high_resolution_clock::now();

    // Calculate and print execution times and speedup
    if (rank == 0) {
        auto duration_serial = duration_cast<milliseconds>(end_serial - start_serial).count();
        auto duration_parallel = duration_cast<milliseconds>(end_parallel - start_parallel).count();
        double speedup = static_cast<double>(duration_serial) / duration_parallel;

        cout << "For Matrix size = " << N << endl;
        cout << "Serial execution time: " << duration_serial << " ms" << endl;
        cout << "Parallel execution time: " << duration_parallel << " ms" << endl;
        cout << "Speedup: " << speedup << endl;
    }

    MPI_Finalize();
    return 0;
}

