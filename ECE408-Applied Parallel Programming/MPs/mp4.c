#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
//#define KERNEL_RADIUS 1
#define TILE_WIDTH 8
#define BLOCK_WIDTH 10
//@@ Define constant memory for device kernel here
__constant__ float M[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int z_out = bz * TILE_WIDTH + tz;
  int y_out = by * TILE_WIDTH + ty;
  int x_out = bx * TILE_WIDTH + tx;

  int z_in = z_out - 1;
  int y_in = y_out - 1;
  int x_in = x_out - 1;

  int xy_size = x_size * y_size;

 // int M_Squ = MASK_WIDTH * MASK_WIDTH;

  __shared__ float input_ds[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

  if((z_in < z_size) && (z_in >= 0) && (y_in < y_size) && (y_in >= 0) && (x_in < x_size) && (x_in >= 0))
    input_ds[tz][ty][tx] = input[z_in * xy_size + y_in * x_size + x_in];
  else
    input_ds[tz][ty][tx] = 0.0f;
  __syncthreads();
  float xyzOutput = 0.0f;

  if((tz < TILE_WIDTH) && (ty < TILE_WIDTH) && (tx < TILE_WIDTH))
  {
    for(int r = 0; r < MASK_WIDTH; r++)
    {
      for(int q = 0; q < MASK_WIDTH; q++)
      {
        for(int p = 0; p <MASK_WIDTH; p++)
        {
          xyzOutput += M[r][q][p] * input_ds[tz + r] [ty + q][tx + p];
        }
      }
    }
  

  if((z_out < z_size) && (y_out < y_size) && (x_out < x_size))
  {
    output[z_out * xy_size + y_out * x_size + x_out] = xyzOutput;
  }
}

  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];

  //actual input
  //for(int i = 0; i < inputLength; i ++)
    //hostInput1[i] = hostInput[i+3];

  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int sizeInput = x_size * y_size * z_size * sizeof(float);
  int sizeOutput = sizeInput;
  cudaMalloc((void **) &deviceInput, sizeInput);
  cudaMalloc((void **) &deviceOutput, sizeOutput);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, sizeInput, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M, hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/8.0),ceil(y_size/8.0),ceil(z_size/8.0));
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, sizeOutput, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
