#include <cstdio>
#include "cpu.h"
#include "common.h"
//#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    if(n==0) return ;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	odata[0]=0;
	for(int i=1;i<n;++i){
		odata[i]=odata[i-1]+idata[i-1];
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"Time used in scan on CPU "<<milliseconds<<" ms"<<std::endl;
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
    int count=0;
	for(int i=0;i<n;++i){
		if(idata[i]!=0) odata[count++]=1;
	}
	return count;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	int *tmp1=new int[n],*tmp2=new int[n];
	for(int i=0;i<n;++i){
		if(idata[i]==0) tmp1[i]=0;
		else tmp1[i]=1;
	}
    scan(n,tmp2,tmp1);
	for(int i=0;i<n;++i){
		if(tmp1[i]!=0){
			odata[tmp2[i]]=1;
		}
	}
	int tmp=tmp2[n-1];
	delete tmp1;
	delete tmp2;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"Time used in compaction on CPU "<<milliseconds<<" ms"<<std::endl;
	
	return tmp;
}

}
}
