/******************************************************************************
 * cr
 * cr            (C) Copyright 2010 The Board of Trustees of the
 * cr                        University of Illinois
 * cr                         All Rights Reserved
 * cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file; you may use more than one kernel if needed

// INSERT KERNEL(S) HERE

//Prefix-scan kernel
__global__ void preScanKernel(float *out, float *in, unsigned size, float *blockSums)
{
    __shared__ float sharedMemory[BLOCK_SIZE * 2];  // Shared memory for storing input data
    int tid = threadIdx.x;  // Thread ID for easy reference
    int offset, iteration;

    int index = 2 * blockIdx.x * BLOCK_SIZE + tid;

    //Load input data into shared memory
    sharedMemory[tid] = (index < size) ? in[index] : 0.0f;
    sharedMemory[tid + BLOCK_SIZE] = (index + BLOCK_SIZE < size) ? in[index + BLOCK_SIZE] : 0.0f;

    //up-sweep(reduction) 
    for (iteration = BLOCK_SIZE, offset = 1; iteration > 0; iteration >>= 1, offset <<= 1) {
        __syncthreads();

        if (tid < iteration) {
            sharedMemory[offset * (2 * tid + 2) - 1] += sharedMemory[offset * (2 * tid + 1) - 1];
        }
    }

    //Store blockSums
    if (tid == 0) {
        if (blockSums != NULL)
            blockSums[blockIdx.x] = sharedMemory[2 * BLOCK_SIZE - 1];
        sharedMemory[2 * BLOCK_SIZE - 1] = 0.0f;
    }

    //Perform down-sweep 
    for (iteration = 1, offset = BLOCK_SIZE; iteration <= BLOCK_SIZE; iteration <<= 1, offset >>= 1) {
        __syncthreads();

        if (tid < iteration) {
            float temp = sharedMemory[offset * (2 * tid + 1) - 1];
            sharedMemory[offset * (2 * tid + 1) - 1] = sharedMemory[offset * (2 * tid + 2) - 1];
            sharedMemory[offset * (2 * tid + 2) - 1] += temp;
        }
    }

    __syncthreads();

    //Write the output data to the memory
    out[index] = sharedMemory[tid];
    out[index + BLOCK_SIZE] = sharedMemory[tid + BLOCK_SIZE];
}

//Kernel for adding blockSums to the result
__global__ void addKernel(float *out, float *blockSums, unsigned size)
{
    int index = 2 * blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Add the block sum to the result
    if (index < size)
        out[index] += blockSums[blockIdx.x];

    if (index + BLOCK_SIZE < size)
        out[index + BLOCK_SIZE] += blockSums[blockIdx.x];
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
    // INSERT CODE HERE
    float *blockSums;
    unsigned totalBlocks;

    //Calculate number of blocks required
    totalBlocks = in_size / (BLOCK_SIZE * 2);
    if (in_size % (BLOCK_SIZE * 2) != 0) {
        totalBlocks++;
    }

    //Set up the grid and block dimensions
    dim3 dim_block(BLOCK_SIZE, 1, 1);
    dim3 dim_grid(totalBlocks, 1, 1);

    //Process if there are multiple blocks
    if (totalBlocks > 1) {
        //Allocate memory for the block sum array
        cudaError_t cuda_ret = cudaMalloc((void**)&blockSums, totalBlocks * sizeof(float));
        if (cuda_ret != cudaSuccess) {
            FATAL("Unable to allocate device memory");
        }

        //Launch the first scan kernel
        preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, blockSums);

        //Recursively scan the block sum array
        preScan(blockSums, blockSums, totalBlocks);

        //Launch the second kernel to add the block sums to the result
        addKernel<<<dim_grid, dim_block>>>(out, blockSums, in_size);

        //Free the block sum array
        cudaFree(blockSums);
    } else {
        //Launch the kernel without the block sum array if there is only one block
        preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, NULL);
    }
}
