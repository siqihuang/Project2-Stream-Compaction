#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	std::vector<int> in;
	for(int i=0;i<n;++i) in.push_back(idata[i]);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	thrust::device_vector<int> dv_in(in.begin(),in.end());
	thrust::device_vector<int> dv_out(n,0);
	thrust::exclusive_scan(dv_in.begin(),dv_in.end(),dv_out.begin());
	std::vector<int> out(dv_out.begin(),dv_out.end());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"Time used in thrust scan on GPU "<<milliseconds<<" ms"<<std::endl;
	for(int i=0;i<n;++i){
		odata[i]=out[i];
	}
}

}
}
