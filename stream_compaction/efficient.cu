#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
namespace StreamCompaction {
namespace Efficient {

// TODO: __global__
__global__ void upSwapOnGPU(int *idata,int step,int n,int newN){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<newN){
		if(step==1&&index>=n) idata[index]=0;
		if((index+1)%(step*2)==0) idata[index]+=idata[index-step];
	}
}

__global__ void downSwapOnGPU(int *idata,int step,int n,int newN){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<newN){
		if(step*2==newN&&index==newN-1) idata[index]=0;
		if((index+1)%(step*2)==0){
			int tmp=idata[index-step];
			idata[index-step]=idata[index];
			idata[index]+=tmp;
		}
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int newN=pow(2,ilog2ceil(n));
	int *dev_idata;
	cudaMalloc((void**)&dev_idata, newN * sizeof(int));
	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockPerGrid=((newN+blockSize-1)/blockSize);
	
	int step=1;
	cudaEventRecord(start);
	while(step<newN){
		upSwapOnGPU<<<blockPerGrid,blockSize>>>(dev_idata,step,n,newN);
		step*=2;
	}
	step/=2;
	while(step!=0){
		downSwapOnGPU<<<blockPerGrid,blockSize>>>(dev_idata,step,n,newN);
		step/=2;
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"Time used in efficient scan on GPU "<<milliseconds<<" ms"<<std::endl;

	cudaMemcpy(odata,dev_idata,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(dev_idata);
}

__global__ void countOne(int n,int *idata,int *odata){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		if(idata[index]!=0) odata[index]=1;
		else odata[index]=0;
	}
}

__global__ void getCompact(int *idata,int *tmp,int *odata,int n){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		if(idata[index]==1) odata[tmp[index]]=idata[index];
	}
}
/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    // TODO
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    int *dev_tmp1,*dev_tmp2,*dev_idata,*dev_odata,*tmp1=new int[n],*tmp2=new int[n];
	cudaMalloc((void**)&dev_tmp1,n*sizeof(int));
	cudaMalloc((void**)&dev_tmp2,n*sizeof(int));
	cudaMalloc((void**)&dev_idata,n*sizeof(int));
	cudaMalloc((void**)&dev_odata,n*sizeof(int));
	cudaMemcpy(dev_idata,idata,n*sizeof(int),cudaMemcpyHostToDevice);

	dim3 blockPerGrid=((n+blockSize-1)/blockSize);
	cudaEventRecord(start);
	countOne<<<blockPerGrid,blockSize>>>(n,dev_idata,dev_tmp1);
	cudaMemcpy(tmp1,dev_tmp1,n*sizeof(int),cudaMemcpyDeviceToHost);
	scan(n,tmp2,tmp1);
	cudaMemcpy(dev_tmp2,tmp2,n*sizeof(int),cudaMemcpyHostToDevice);
	getCompact<<<blockPerGrid,blockSize>>>(dev_tmp1,dev_tmp2,dev_odata,n);
	cudaMemcpy(odata,dev_odata,n*sizeof(int),cudaMemcpyDeviceToHost);
	int count=tmp2[n-1]+tmp1[n-1];

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	delete tmp1;
	delete tmp2;
	cudaFree(dev_tmp1);
	cudaFree(dev_tmp2);
	cudaFree(dev_idata);
	cudaFree(dev_odata);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"Time used in compaction on GPU "<<milliseconds<<" ms"<<std::endl;

	return count;
}

}
}
