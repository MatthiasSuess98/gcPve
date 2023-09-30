#ifndef GCPVE_21_L1_CACHE_BENCHMARK_CUH
#define GCPVE_21_L1_CACHE_BENCHMARK_CUH

#include "00-Main.cu"
#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"


__shared__ unsigned int saveValue[4];


/**
 * Kernel benchmark which analyzes the load time of different memory types.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 * @param deviceLoad The load which will be loaded.
 * @param deviceTime The time differences of the loading process.
 * @param i The current multiprocessor.
 * @param j The current experimental warp.
 */
__global__ void benchmark(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives, unsigned int *deviceLoad, float *deviceTime, int i, int j) {

    int mulp;
    int lane;

    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));

    if (mulp == i) {
        for (int k = 0; k < info.warpSize; k++) {
            if (lane == k) {

                unsigned long long endTime;
                unsigned long long startTime;

                unsigned int value;
                unsigned int *ptr;
                unsigned int valueArray[1024];
                __shared__ unsigned int load[1024];

                // L1 benchmark.
                value = 0;
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    ptr = deviceLoad + value;
                    asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
                }
                saveValue[0] = value;
                value = 0;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    ptr = deviceLoad + value;
                    asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
                }
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));
                saveValue[0] = value;
                deviceTime[k + (0 * info.warpSize)] = ((float) (endTime - startTime)) / ((float) prop.numberOfTrialsBenchmark);

                // SM benchmark.
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    valueArray[l] = 0;
                }
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    load[l] = deviceLoad[l];
                }
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    valueArray[l] = load[l] + valueArray[l];
                }
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));
                saveValue[1] = valueArray[1023];
                deviceTime[k + (1 * info.warpSize)] = ((float) (endTime - startTime)) / ((float) prop.numberOfTrialsBenchmark);

                // L2 benchmark.
                value = 0;
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    ptr = deviceLoad + value;
                    asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
                }
                saveValue[2] = value;
                value = 0;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    ptr = deviceLoad + value;
                    asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
                }
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));
                saveValue[2] = value;
                deviceTime[k + (2 * info.warpSize)] = ((float) (endTime - startTime)) / ((float) prop.numberOfTrialsBenchmark);

                // GM benchmark.
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    valueArray[l] = 0;
                }
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));
                for (int l = 0; l < prop.numberOfTrialsBenchmark; l++) {
                    valueArray[l] = deviceLoad[l] + valueArray[l];
                }
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));
                saveValue[3] = valueArray[1023];
                deviceTime[k + (3 * info.warpSize)] = ((float) (endTime - startTime)) / ((float) prop.numberOfTrialsBenchmark);
            }
        }
    }
}


#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

//FINISHED

