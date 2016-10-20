// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *sum, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  __shared__ float XY[BLOCK_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) 
    XY[threadIdx.x] = input[i];
  else
    XY[threadIdx.x] = 0;
 
  for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if(index < BLOCK_SIZE)
    {
      XY[index] += XY[index - stride];
    }
  }
  
  for(int stride = BLOCK_SIZE/4; stride > 0; stride /= 2)
  {
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if(index + stride < BLOCK_SIZE)
    {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  if(i < len) 
    output[i] = XY[threadIdx.x];
  
  if(len > blockDim.x)
  {
    if(threadIdx.x == blockDim.x - 1)
    {
      sum[blockIdx.x] = XY[threadIdx.x];
    }
  }
}


__global__ void postscan(float *input1, float *input2, float *output, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len)
  {
    if(blockIdx.x > 0)
      output[i] = input1[i] + input2[blockIdx.x - 1];
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *sum;
  int numsum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allo cating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&sum, numElements * sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(sum, 0, ceil(numElements/512.0) * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/512.0),1,1);
  dim3 blockDim(BLOCK_SIZE,1,1);
  numsum = ceil(numElements / 512.0);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, blockDim>>>(deviceInput, deviceOutput, sum, numElements);
  scan<<<dimGrid, blockDim>>>(sum, sum, sum, numsum);
  postscan<<<dimGrid, blockDim>>>(deviceOutput, sum, deviceOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(sum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}