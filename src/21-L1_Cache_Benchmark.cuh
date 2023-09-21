#ifndef GCPVE_21_L1_CACHE_BENCHMARK_CUH
#define GCPVE_21_L1_CACHE_BENCHMARK_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

/**
 * The L1 benchmark which uses the small data collection.
 * @param ptr The small data collection where the data of the benchmark is stored.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
__global__ void smallL1Benchmark(SmallDataCollection *ptr, GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    int pos = (blockIdx.x * info.warpSize) + threadIdx.x;
    int mulp;
    int warp;
    int lane;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));
    mulp = (*ptr).mulp[pos];
    warp = (*ptr).warp[pos];
    lane = (*ptr).lane[pos];
    unsigned int startTime;
    unsigned int endTime;
    //int load = prop.load;
    unsigned int* load;
    unsigned int zero = 0;
    for (int preparationLoop = 0; preparationLoop < prop.numberOfTrialsBenchmark; preparationLoop++) {
        asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(zero) : "l"(load) : "memory");
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(startTime));
    for (int measureLoop = 0; measureLoop < prop.numberOfTrialsBenchmark; measureLoop++) {
        asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(zero) : "l"(load) : "memory");
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(endTime));
    (*ptr).time[pos] = ((float) (endTime - startTime)) / ((float) prop.numberOfTrialsBenchmark);
    printf("%.9g\n", (*ptr).time[pos]);
}

/**
 * Launches the L1 benchmarks with small data collection.
 * @param ptr The small data collection where the data of the benchmarks is stored.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
void launchSmallL1Benchmarks(SmallDataCollection *ptr, GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    smallL1Benchmark<<<derivatives.smallNumberOfBlocks, info.warpSize>>>(ptr, info, prop, derivatives);
    cudaDeviceSynchronize();
}

#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

//FINISHED

