/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define SIMPLE

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
//conditional block for SIMPLE mode
#ifdef SIMPLE 
    __shared__ float sdata[2*BLOCK_SIZE]; //shared memory for storeing two segments of input
    int tid = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + tid; //calculate global index

    // load two elements into shared memory
    sdata[tid] = ((i < size)? in[i]: 0.0f);
    sdata[tid+BLOCK_SIZE] = ((i + BLOCK_SIZE<size)? in[i+BLOCK_SIZE] :0.0f);

    //reduction loop for combining elements 
    for (int s = 1; s < BLOCK_SIZE<<1; s <<=1) { //stride doubles each iteration
        __syncthreads();
        if (tid % s == 0) {//tid is dividible by stride
            sdata[2*tid] += sdata[2*tid+s]; //sum paris of elements with stride spacing
        }
    }

#else //non-SIMPLE mode reduction
    __shared__ float sdata[BLOCK_SIZE];
    int i = 2 * blockIdx.x * blockDim.x + tid;

    //two elements into a single shared memory 
    sdata[tid] = ((i < size)? in[i]: 0.0f)+ ((i + BLOCK_SIZE<size)? in[i+BLOCK_SIZE] :0.0f);

    //reduction loop
    for (int s = BLOCK_SIZE>>1; s > 0; s >>=1) {  //stride halves each iteration
        __syncthreads();
        if (tid < s) { //tid within stride
            sdata[tid] += sdata[tid+s]; //sum pairs of elements seperated by stride
        }
    }


#endif //write the result in global memory
    if(tid == 0) out[blockIdx.x] = sdata[0];
}
