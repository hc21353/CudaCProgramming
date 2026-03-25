/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

// Feel free to use other numbers for best performance
#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C)
{

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    __shared__ float subTileA[TILE_SIZE][TILE_SIZE]; //shared memory tile for A
    __shared__ float subTileB[TILE_SIZE][TILE_SIZE]; //shared memory tile for B

    //thread ID in block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //identifu the row and column of the C element to work on
    int row = blockIdx.y * blockDim.y + ty; 
    int col = blockIdx.x * blockDim.x + tx;

    float Cval = 0.0f; //save the value to compute one element of matrix C

    // Loop over the A and B tiles required to compute the C element
    for (int tileIdx = 0; tileIdx < (k-1)/TILE_SIZE + 1; ++tileIdx) //adds an extra iteration for partial tiles if k is not a multiple of TILE_SIZE
    {
        //load A tile in shared memory
        if (row < m && tileIdx * TILE_SIZE + tx < k) {
            subTileA[ty][tx] = A[row * k + tileIdx * TILE_SIZE + tx];
        } else {
            subTileA[ty][tx] = 0;
        }

        //load B tile in shared memory
        if(col < n && tileIdx * TILE_SIZE + ty < k) {
            subTileB[ty][tx] = B[(tileIdx * TILE_SIZE + ty) * n + col];
        } else {   
            subTileB[ty][tx] = 0;
        }

        __syncthreads(); //hold on untill all the treads load the data

        //do matrix multiplication whithin the tile
        if (row < m && col < n) {
            for (int i = 0; i < TILE_SIZE; ++i) {
                Cval += subTileA[ty][i] * subTileB[i][tx];
            }
        }
    
        __syncthreads(); //wait until tile calculation done
    }
        //save result in C matrix
        if (row < m && col < n) {
            C[row * n + col] =  Cval;
        }
    

}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, int testRound)
{
    if ((transa != 'N') && (transa != 'n'))
    {
        printf("unsupported value of 'transa'\n");
        return;
    }

    if ((transb != 'N') && (transb != 'n'))
    {
        printf("unsupported value of 'transb'\n");
        return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10))
    {
        printf("unsupported value of alpha\n");
        return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10))
    {
        printf("unsupported value of beta\n");
        return;
    }

    // Initialize thread block and kernel grid dimensions ----------------------
    // INSERT CODE HERE
    int BLOCK_SIZE = TILE_SIZE;
    dim3 dimGrid((n - 1)/BLOCK_SIZE + 1, (m - 1)/BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);




    for (int i = 0; i < testRound; i++) {
        // Invoke CUDA kernel --------------------------------------------------
        // INSERT CODE HERE
        mysgemm<<<dimGrid, dimBlock>>> (m,n,k,A,B,C);


        cudaDeviceSynchronize();
    }
}
