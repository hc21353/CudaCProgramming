/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
	// Calculate global thread index based on the block and thread indices ----
	//INSERT KERNEL CODE HERE
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Use global index to determine which elements to read, add, and write ---
	//INSERT KERNEL CODE HERE, BE CAREFUL FOR CORNER CASE!!!
	if (i < n) {
		C[i] = A[i] + B[i];
	}
	
}

__global__ void image2grayKernel(float *in, float *out, int height, int width)
{
	// Calculate global thread index based on the block and thread indices ----
	//INSERT KERNEL CODE HERE
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	
	// Use global index to determine which elements to read, add, and write ---
	//INSERT KERNEL CODE HERE, BE CAREFUL FOR CORNER CASE!!!
	if (col < width && row <height) {
		int greyOffset = row * width + col;
		int rgbOffset = 3 * greyOffset;
		float r = in[rgbOffset]; //red value for pixel
		float g = in[rgbOffset + 1]; //green value for pixel
		float b = in[rgbOffset + 2]; //blue value for pixel
		
		//out[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b; //convert rgb to grayscale (가중치 평균 사용)
		out[greyOffset] = 0.144f*r + 0.587f*g + 0.299f*b;
	}
	
}