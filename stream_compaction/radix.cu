#include <cuda.h>
#include <cuda_runtime.h>
#include "radix.h"
#include "efficient.h"
#include "common.h";
#include <iostream>

namespace StreamCompaction {
namespace Radix {

void scan(int n, int *odata, const int *idata){
	StreamCompaction::Efficient::scan(n,odata,idata);
}

__device__ int getDigit(int n,int pos){
	int result=0;
	for(int i=0;i<pos;++i){
		result=n%2;
		n/=2;
	}
	return result;
}

__global__ void getDigits(int n, int *idata, int *odata,int pos){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		odata[index]=getDigit(idata[index],pos);
	}
}

__global__ void Reverse(int n,int *idata,int *odata){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		odata[index]=1-idata[index];
	}
}

__global__ void getT(int n,int *idata,int *odata,int totalFalse){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		odata[index]=index-idata[index]+totalFalse;
	}
}

__global__ void getPos(int n,int *dev_b,int *dev_t,int *dev_f,int *odata){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		odata[index]=dev_b[index]*dev_t[index]+(1-dev_b[index])*dev_f[index];
	}
}

__global__ void switchPos(int n,int *idata, int *odata, int *dev_d){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n){
		odata[dev_d[index]]=idata[index];
	}
}

void radix(int n, int *odata, const int *idata){
	int num=ilog2ceil(n-1);
	int *dev_idata,*dev_odata,*dev_b,*dev_e,*dev_f,*dev_t,*dev_d;
	int *host_f=new int[n],*host_e=new int[n];
	cudaMalloc((void**)&dev_idata, n*sizeof(int));
	cudaMalloc((void**)&dev_odata, n*sizeof(int));
	cudaMalloc((void**)&dev_b, n*sizeof(int));
	cudaMalloc((void**)&dev_e, n*sizeof(int));
	cudaMalloc((void**)&dev_f, n*sizeof(int));
	cudaMalloc((void**)&dev_t, n*sizeof(int));
	cudaMalloc((void**)&dev_d, n*sizeof(int));
	cudaMemcpy(dev_idata,idata,n*sizeof(int),cudaMemcpyHostToDevice);

	dim3 blockPerGrid((n+blockSize-1)/blockSize);
	for(int i=1;i<=num;++i){
		if(i%2==1) getDigits<<<blockPerGrid,blockSize>>>(n,dev_idata,dev_b,i);
		else getDigits<<<blockPerGrid,blockSize>>>(n,dev_odata,dev_b,i);
		Reverse<<<blockPerGrid,blockSize>>>(n,dev_b,dev_e);
		cudaMemcpy(host_e,dev_e,n*sizeof(int),cudaMemcpyDeviceToHost);
		scan(n,host_f,host_e);
		cudaMemcpy(dev_f,host_f,n*sizeof(int),cudaMemcpyHostToDevice);
		int totalFalse=host_e[n-1]+host_f[n-1];
		getT<<<blockPerGrid,blockSize>>>(n,dev_f,dev_t,totalFalse);
		getPos<<<blockPerGrid,blockSize>>>(n,dev_b,dev_t,dev_f,dev_d);
		if(i%2==1) switchPos<<<blockPerGrid,blockSize>>>(n,dev_idata,dev_odata,dev_d);
		else switchPos<<<blockPerGrid,blockSize>>>(n,dev_odata,dev_idata,dev_d);
	}

	if(num%2==1) cudaMemcpy(odata,dev_odata,n*sizeof(int),cudaMemcpyDeviceToHost);
	else cudaMemcpy(odata,dev_idata,n*sizeof(int),cudaMemcpyDeviceToHost);

	cudaFree(dev_idata);
	cudaFree(dev_odata);
	cudaFree(dev_b);
	cudaFree(dev_t);
	cudaFree(dev_d);
	cudaFree(dev_f);
	cudaFree(dev_e);
}

}
}
