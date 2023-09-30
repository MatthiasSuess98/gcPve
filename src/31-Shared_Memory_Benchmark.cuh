#ifndef GCPVE_31_SHARED_MEMORY_BENCHMARK_CUH
#define GCPVE_31_SHARED_MEMORY_BENCHMARK_CUH

#include "00-Main.cu"
#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

/**
 *
 */
__global__ void smallSMBenchmark(unsigned int *deviceLoad, float *deviceTime, int i, int k) {

    int mulp;
    int warp;
    int lane;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));
    if ((mulp == i) && (warp == (k % 5))) {

        unsigned long long endTime;
        unsigned long long startTime;


    }
}

#endif //GCPVE_31_SHARED_MEMORY_BENCHMARK_CUH

//FINISHED

