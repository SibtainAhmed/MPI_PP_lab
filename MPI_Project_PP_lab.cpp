// MPI_Project_PP_lab.cpp : This file contains the 'main' function. Program execution begins and ends there.
//





#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <cstdlib>

//#include <unistd.h>
//#include <iostream.h>


int main(int argc, char** argv)
{
	int mynode, totalnodes;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
	printf("Hello world from process %d of %d", mynode, totalnodes);
	MPI_Finalize();
}



//MPI_SCAN
#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_data = rank + 1;  // Each process has its rank as its data

    int result;  // This will hold the partial result in each process

    // Compute the running sum of all local_data and store it in result for each process
    MPI_Reduce(&local_data, &result, 1, MPI_INT, MPI_SUM,0, MPI_COMM_WORLD);

    //if (rank == 0) {
    //    printf("result of sum = %d", result);
    //}
    //MPI_Reduce(&local_data, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);



    printf("Process % d has running sum % d", rank, result);

    //cout << "Process " << rank << " has running sum " << result << endl;

    MPI_Finalize();
    return 0;
}




// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file


//================================== Second Evaluation ====================================


#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int stateSum = rank+1;

	int prefixSum;

	MPI_Scan(&stateSum, &prefixSum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	printf("Prefix sum at process %d = % d", rank, prefixSum);

	MPI_Finalize();
	return 0;
}









#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
	int rank, totalnodes;
	int datasize = 4;
	int root = 0;
	double* broadcastArray = new double[datasize];

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == root) {
		for (int i = 0;i < datasize;i++) {
			broadcastArray[i] = (double) i+1;
		 }
	}
	MPI_Bcast(broadcastArray, datasize, MPI_DOUBLE, root, MPI_COMM_WORLD);

	printf("At process %d value of brodcasted variable is = [ %f, %f, %f, %f ]", rank, broadcastArray[0], broadcastArray[1], broadcastArray[2], broadcastArray[3]);

	MPI_Finalize();
	return 0;
}












#include <iostream>
#include <mpi.h>
#include <vector>
#include <string>
#include <numeric>
int main(int argc, char** argv)
{

	
	//Ex:1
	MPI_Init(&argc, &argv);
	//cout << "Hello World!" << endl;
	printf("Hello World!");
	MPI_Finalize();
	

	//Ex2
	int mynode, totalnodes, len;
	char myname[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
	MPI_Get_processor_name(myname, &len);
	printf("Hello world from process %s with rank %d of %d", myname, mynode, totalnodes);
	MPI_Finalize();


	//Ex3
	int mynode, totalnodes;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
	//MPI_Get_processor_name(myname, &len);
	if (mynode % 2) {
	    printf("I am ODD; processId = %d", mynode);
	}
	else{
	    printf("I am EVEN; processId = %d", mynode);
	}
	MPI_Finalize();


	//=========================  Lab 8  =========================

	//Excercise#1
	
		//Example#1
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		printf("example#1 can't be run on less than 5 processes !!!!");
	}

	const int source = 2;
	const int dest = 4;
	const int tag = 0;

	if (rank == source) {
		const char* message = "Hello from process 2";
		MPI_Send(message, strlen(message) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
		std::cout << "Process " << rank << " sent message: " << message << std::endl;
	}
	else if (rank == dest) {
		// Buffer to receive the message
		char buffer[100];
		MPI_Recv(buffer, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::cout << "Process " << rank << " received message: " << buffer << std::endl;
	}

	MPI_Finalize();

	return 0;
	

	
	//Example#2
	int mynode, totalnodes;
	int sum, startval, endval, accum;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
	sum = 0;
	startval = 1000 * mynode / totalnodes + 1;
	endval = 1000 * (mynode + 1) / totalnodes;
	for (int i = startval;i <= endval;i = i + 1)
		sum = sum + i;
	if (mynode != 0) {
		printf("sending sum %d from process %d of number %d to %d", sum, mynode, startval, endval);
		MPI_Send(&sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
	}
	else{
		for (int j = 1;j < totalnodes;j = j + 1)
		{

			MPI_Recv(&accum, 1, MPI_INT, j, 1, MPI_COMM_WORLD, &status);

			sum = sum + accum;
		}
		printf("calculating sum at process %d from number %d to %d\n",  mynode, startval, endval);
		std::cout << "The sum from 1 to 1000 is: " << sum << std::endl;
	}

	MPI_Finalize();

	
	
    //Example3

	int i;
	int nitems = 4;
	int mynode, totalnodes;
	MPI_Status status;
	int* array;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
	array = new int[nitems];
	if (mynode == 0)
	{
		for (i = 0;i < nitems;i++)
			array[i] = (int)i;
	}
	if (mynode == 0)
		for (i = 1;i < totalnodes;i++)
			MPI_Send(array, nitems, MPI_INT, i, 1, MPI_COMM_WORLD);

	else {

		MPI_Recv(array, nitems, MPI_INT, 0, 1, MPI_COMM_WORLD,&status);
	
		//std::cout << "Processor " << mynode;
		//std::cout << ": array[" << i << "] = " << array[i] << std::endl;
		printf("Process %d: ", mynode);
		printf("Array[%d,%d,%d,%d]", array[0], array[1], array[2], array[3]);
	}

	MPI_Finalize();
	

//
//// Ex2
	// Initialize the MPI environment
MPI_Init(&argc, &argv);

// Get the rank of the process
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Get the number of processes
int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Each process has an array buffer to send
const int array_size = 4;
std::vector<int> send_buffer(array_size, rank); // Fill the buffer with the rank value for demonstration


if (rank == 0) {
	// Root process receives data from all other processes
	std::vector<int> recv_buffer(array_size);
	for (int i = 1; i < size; ++i) {
		MPI_Recv(recv_buffer.data(), array_size, MPI_INT, i, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::cout << "Process 0 received data from process " << i << ": ";
			std::cout << "[ ";
		for (int val : recv_buffer) {
			std::cout << val << ", ";

		}
			std::cout << "]";
		std::cout << std::endl;
	}
}
else {
	// Non-root processes send their data to the root process
	MPI_Send(send_buffer.data(), array_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	std::cout << "Process " << rank << " sent data to process 0 " << std::endl;
}

// Finalize the MPI environment
MPI_Finalize();

return 0;



// Ex3

// Initialize the MPI environment
MPI_Init(&argc, &argv);

// Get the rank of the process
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Get the number of processes
int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);

int res = 3;
int send_value = res + 1;
int recv_value;

if (rank == 0) {
	// First process only sends to the right
	MPI_Send(&send_value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
	printf("Process %d sent number=%d to process %d\n", rank, send_value, rank + 1);
}
else if (rank == size - 1) {
	// Last process only receives from the left
	MPI_Recv(&recv_value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("Process %d received number=%d from process %d\n", rank, recv_value, rank - 1);
}
else {
	// Other processes receive from the left and send to the right
	MPI_Recv(&recv_value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("Process %d received number=%d from process %d\n", rank, recv_value, rank - 1);

	// Update send_value with received value for demonstration
	send_value = recv_value + 1;

	MPI_Send(&send_value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
	printf("Process %d sent number=%d to process %d\n", rank, send_value, rank + 1);
}

// Finalize the MPI environment
MPI_Finalize();

return 0;




// Ex4

int a, b;
// Initialize the MPI environment
MPI_Init(&argc, &argv);

// Get the rank of the process
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Get the number of processes
int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);


if (rank == 0) {
	a = rank;
	// First process only sends to the right
	MPI_Send(&a, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
	printf("At process %d Prefix sum = %d\n", rank, a);
}
else if (rank == size - 1) {
	// Last process only receives from the left
	MPI_Recv(&b, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	a = b + rank;
	printf("At process %d Prefix sum = %d\n", rank, a);
}
else {
	// Other processes receive from the left and send to the right
	MPI_Recv(&b, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// Update send_value with received value for demonstration
	a = b + rank;
	printf("At process %d Prefix sum = %d\n", rank, a);
	MPI_Send(&a, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
}

// Finalize the MPI environment
MPI_Finalize();

return 0;




//================================== Lab 9 ========================== 

// Ex 1
const int total_iterations = 100;

MPI_Init(&argc, &argv);

// Get the number of processes
int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Get the rank of the process
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int iterations_per_process = total_iterations / size;

int start_iteration = rank * iterations_per_process;
int end_iteration = start_iteration + iterations_per_process;

	std::cout << "Process " << rank << " handling iteration from "
		<< start_iteration << " to " << end_iteration << std::endl;
for (int i = start_iteration; i < end_iteration; ++i) {
	continue;
}
std::cout << "Process " << rank 
<< " waiting for other processes to complete " << std::endl;
MPI_Barrier(MPI_COMM_WORLD);

if (rank == 0) {
	printf("All process finished and passed MPI_Barrier !");
}
// Finalize the MPI environment
MPI_Finalize();

return 0;




// Ex 2
MPI_Init(&argc, &argv);

int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);

MPI_Barrier(MPI_COMM_WORLD);
if (rank == 0) {
	std::cout << "All processes are alive and running. :-)" << std::endl;
}

int send_value = rank+1000;
int recv_value;
int next_rank = (rank + 1) % size;
int prev_rank = (rank - 1 + size) % size;

MPI_Send(&send_value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
MPI_Recv(&recv_value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

std::cout << "Process " << rank << " sent " << send_value << " to process " << next_rank
<< "\n and received " << recv_value << " from process " << prev_rank << std::endl;

MPI_Finalize();

return 0;


// Ex 3

MPI_Init(&argc, &argv);

int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);

char work;

if (rank == 0) {
	work = 'R';
	std::cout << "Process 0 is broadcasting the string = " << work << std::endl;
}

	MPI_Bcast(&work, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

std::cout << "Process " << rank << " received the string = " << work << std::endl;

MPI_Finalize();

return 0;




// Ex 4

MPI_Init(&argc, &argv);

int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);

const int vector_size = 16; 
std::vector<int> vector(vector_size);

if (rank == 0) {
	for (int i = 0; i < vector_size; ++i) {
		vector[i] = i + 1;  
	}
}
int sub_vector_size = vector_size / size;
std::vector<int> sub_vector(sub_vector_size);

MPI_Scatter(vector.data(), sub_vector_size, MPI_INT,
sub_vector.data(), sub_vector_size, MPI_INT, 0, MPI_COMM_WORLD);

int partial_sum = 0;
for (int i = 0; i < sub_vector_size; ++i) {
	partial_sum += sub_vector[i];
}
printf("Process %d calculate partial sum = %d", rank, partial_sum);

std::vector<int> partial_sums(size);
MPI_Gather(&partial_sum, 1, MPI_INT, partial_sums.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

if (rank == 0) {
	int total_sum = 0;
	for (int i = 0; i < size; ++i) {
		total_sum += partial_sums[i];
	}
	std::cout << "\nTotal sum: " << total_sum << std::endl;
}

// Finalize the MPI environment
MPI_Finalize();

return 0;



// Ex 5

MPI_Init(&argc, &argv);

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int local_data = rank + 1; 

int result;

MPI_Scan(&local_data, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

std::cout << "Process " << rank << " has running sum " << result << std::endl;

MPI_Finalize();

}

