/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
    /********************************************************************
    Determine input and output indexes of each thread
    Load a tile of the input image to shared memory
    Apply the filter on the input image tile
    Write the compute values to the output image at the correct indexes
    ********************************************************************/

    // INSERT KERNEL CODE HERE

    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE]; //shared memory tile

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_o = blockIdx.x*TILE_SIZE+tx; //output index
    int row_o = blockIdx.y*TILE_SIZE+ty;

    int col_i = col_o-(FILTER_SIZE/2); //input index; radius = mask_width/w
    int row_i = row_o-(FILTER_SIZE/2);

    float output = 0.0f;

    //check the input indices are valid
    if(row_i >= 0 && row_i < N.height && col_i >= 0 && col_i < N.width) {
        tile[ty][tx] = N.elements[row_i*N.width+col_i]; //load input to shared memory
    } else {
        tile[ty][tx] = 0.0f; //out of bounds
    }

    __syncthreads();

    if(tx < TILE_SIZE && ty < TILE_SIZE) { //current thread is within tile size
        //apply filter
        for(int i=0; i < FILTER_SIZE; i++) { 
            for (int j=0; j<FILTER_SIZE; j++) {
                output += M_c[i][j]*tile[i+ty][j+tx]; //accumulate conv result
            }
        }
        if (row_o < P.height && col_o < P.width) { //output indices are valid
            P.elements[row_o*P.width+col_o] = output; //store result in the output image
        }
    }
}
