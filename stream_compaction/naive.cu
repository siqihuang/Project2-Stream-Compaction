#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
__global__ void scanOnGPU(int n, int *odata, int *idata,int step){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		if(index>=step) odata[index]=idata[index]+idata[index-step];
		else odata[index]=idata[index];
	}
}

void scan(int n, int *odata, const int *idata) {
    // TODO
    int step=1,count=0;
	int *dev_odata,*dev_idata;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&dev_odata, n * sizeof(int));
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockPerGrid=((n+blockSize-1)/blockSize);
	cudaEventRecord(start);
	while(step<n){
		if(count%2==0) scanOnGPU<<<blockPerGrid,blockSize>>>(n,dev_odata,dev_idata,step);
		else scanOnGPU<<<blockPerGrid,blockSize>>>(n,dev_idata,dev_odata,step);
		count++;
		step*=2;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	if(count%2==1) cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
	else cudaMemcpy(odata, dev_idata, n*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=n-1;i>0;--i){
		odata[i]=odata[i-1];
	}
	odata[0]=0;
	cudaFree(dev_odata);
	cudaFree(dev_idata);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"Time used in naive scan on GPU "<<milliseconds<<" ms"<<std::endl;
}

}
}
