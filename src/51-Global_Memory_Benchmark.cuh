#ifndef GCPVE_51_GLOBAL_MEMORY_BENCHMARK_CUH
#define GCPVE_51_GLOBAL_MEMORY_BENCHMARK_CUH

#include "00-Main.cu"
#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

/**
 *
 */
__global__ void smallGMBenchmark(unsigned int *deviceLoad, float *deviceTime, int i, int k) {

    int mulp;
    int warp;
    int lane;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (mulp == i) {

        unsigned long long endTime;
        unsigned long long startTime;

        unsigned int value[1024];



        for (int j = 0; j < 1024; j++) {
            value[j] = 0;
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));
        for (int j = 0; j < 1024; j++) {
            for (int m = 0; m < 1024; m++) {
                value[m] = deviceLoad[m] + value[m];
            }
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));
        saveValue[3] = value[0];
        deviceTime[lane] = ((float) (endTime - startTime)) / 1024;
    }
}


#endif //GCPVE_51_GLOBAL_MEMORY_BENCHMARK_CUH

//FINISHED

