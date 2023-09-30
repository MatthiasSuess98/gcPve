#ifndef GCPVE_41_L2_CACHE_BENCHMARK_CUH
#define GCPVE_41_L2_CACHE_BENCHMARK_CUH

#include "00-Main.cu"
#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"


/**
 *
 */
__global__ void smallL2Benchmark(unsigned int *deviceLoad, float *deviceTime, int i, int k) {

    int mulp;
    int warp;
    int lane;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));
    if ((mulp == i) && (warp == 0)) {

        unsigned long long endTime;
        unsigned long long startTime;

        unsigned int value = 0;
        unsigned int *ptr;

        //Load Data in L2 cache.
        for (int j = 0; j < 1024; j++) {
            ptr = deviceLoad + value;
            asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
        }


        //Perform benchmark.
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));

        for (int j = 0; j < 1024; j++) {
            ptr = deviceLoad + value;
            asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
        }


        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));

        saveValue[2] = value;

        deviceTime[lane] = ((float) (endTime - startTime)) / 1024;
    }
}


#endif //GCPVE_41_L2_CACHE_BENCHMARK_CUH

//FINISHED

