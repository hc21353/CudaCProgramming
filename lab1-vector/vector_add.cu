/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include <stdio.h>
#include "support.h"
#include "kernel.cu"

void verify(float *A, float *B, float *C, int n)
{
	const float relativeTolerance = 1e-6;
	for (int i = 0; i < n; i++)
	{
		float sum = A[i] + B[i];
		float relativeError = (sum - C[i]) / sum;
		if (relativeError > relativeTolerance ||
			relativeError < -relativeTolerance)
		{
			printf("TEST FAILED\n\n");
			exit(0);
		}
	}

	printf("TEST PASSED\n\n");
}

int main(int argc, char **argv)
{
	Timer timer;
	cudaError_t cuda_ret;

	// Initialize host variables ----------------------------------------------

	printf("\nSetting up the problem...");
	fflush(stdout);
	startTime(&timer);

	unsigned int n;
	if (argc == 1)
	{
		n = 10000;
	}
	else if (argc == 2)
	{
		n = atoi(argv[1]);
	}
	else
	{
		printf("\n    Invalid input parameters!"
			"\n    Usage: ./vector_add       # Vector of size 10,000 is used"
			"\n    Usage: ./vector_add <m>   # Vector of size m is used"
			"\n");
		exit(0);
	}

	float *A_h = (float*) malloc(sizeof(float) *n);
	for (unsigned int i = 0; i < n; i++)
	{
		A_h[i] = (rand() % 100) / 100.00;
	}

	float *B_h = (float*) malloc(sizeof(float) *n);
	for (unsigned int i = 0; i < n; i++)
	{
		B_h[i] = (rand() % 100) / 100.00;
	}

	float *C_h = (float*) malloc(sizeof(float) *n);

	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));
	printf("    Vector size = %u\n", n);

	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	//allocate GPU memory space for vector A_h, B_h, and C_h bu using cudaMalloc
	float *A_d, *B_d, *C_d; //device copies of A,B,C
	int size = n*sizeof(float); //size of each vector
	cudaMalloc((void**)&A_d, size); // allocate gpu memory as the size of vector A
	cudaMalloc((void**)&B_d, size); // allocate gpu memory as the size of vector B
	cudaMalloc((void**)&C_d, size); // allocate gpu memory as the size of vector C
	
	
	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy host variables to device ------------------------------------------
	printf("Copying data from host to device...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice); //compy the data from A_h to A_d
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice); //compy the data from B_h to B_d

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE 
	float BLOCK_SIZE = 256.0; //tested 128/256/512 and 512 works best in my case
	vecAddKernel<<<ceil(n/BLOCK_SIZE), BLOCK_SIZE>>>(A_d, B_d, C_d, n);

	cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy device variables from host ----------------------------------------
	printf("Copying data from device to host...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	cudaMemcpy(C_h, C_d, n*sizeof(float), cudaMemcpyDeviceToHost); //copy the result from C_d to C_h

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Verify correctness -----------------------------------------------------
	printf("Verifying results...");
	fflush(stdout);
	verify(A_h, B_h, C_h, n);


	// Free memory ------------------------------------------------------------
	free(A_h);
	free(B_h);
	free(C_h);
	//INSERT CODE HERE
	cudaFree(A_d); //free device memory
	cudaFree(B_d);
	cudaFree(C_d);
	

	return 0;

}