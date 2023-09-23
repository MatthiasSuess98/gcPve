#ifndef GCPVE_21_L1_CACHE_BENCHMARK_CUH
#define GCPVE_21_L1_CACHE_BENCHMARK_CUH

#include <vector>

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

/**
 * The L1 benchmark which uses the data collection.
 * @param ptr The data collection where the data of the benchmark is stored.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
__global__ void l1Benchmark(DataCollection * ptr, int requiredLane, unsigned int * load, GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    int mulp;
    int warp;
    int lane;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));
    if ((lane == requiredLane) && ((warp == 0) || (warp == 1) || (warp == 2) || (warp == 3))) {
        int pos = (((mulp * derivatives.numberOfBlocksPerMulp) + blockIdx.x) * info.warpSize) + lane;
        (*ptr).mulp[pos] = mulp;
        (*ptr).warp[pos] = warp;
        (*ptr).lane[pos] = lane;
        long long int startTime[1024];
        long long int endTime[1024];
        unsigned int value = 0;
        for (int preparationLoop = 0; preparationLoop < 1024; preparationLoop++) {
            //value = 0;
            asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(load) : "memory");
        }
        unsigned int counter = 0;
        unsigned int final = 0;
        for (int mainLoop = 0; mainLoop < 1024; mainLoop++) {
            for (int measureLoop = 0; measureLoop < 1024; measureLoop++) {
                value = 0;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime[measureLoop]));
                asm volatile ("ld.ca.u32 %0, [%1];" : "=r"(value) : "l"(load) : "memory");
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime[measureLoop]));
                //asm volatile ("add.u32 %0, %1, %2;" : "=r"(value) : "r"(value), "r"(2));
            }
            counter = 0;
            for (int i = 0; i < 1024; i++) {
                counter = counter + (endTime[i] - startTime[i]);
            }
            final = final + counter;
        }
        (*ptr).time[pos] = ((float) final) / ((float) (1024 * 1024));
    }
}


/**
 * Launches the L1 benchmarks with data collection.
 * @param ptr The data collection where the data of the benchmarks is stored.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
DataCollection launchL1Benchmarks(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    DataCollection data;
    for (int i = 0; i < derivatives.collectionSize; i++) {
        int iniMulp = 0;
        data.mulp.push_back(iniMulp);
        int iniWarp = 0;
        data.warp.push_back(iniWarp);
        int iniLane = 0;
        data.lane.push_back(iniLane);
        long long int iniTime = 0;
        data.time.push_back(iniTime);
    }
    DataCollection *hostPtr;
    //ptr = &data;
    //cudaMallocManaged(&ptr, sizeof(benchCollection));
    cudaHostAlloc(&hostPtr, sizeof(data));
    DataCollection *devicePtr;
    cudaMalloc(&devicePtr, sizeof(data));
    cudaMemcpy(devicePtr, hostPtr, sizeof(data), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
        unsigned int *hostLoad;
        hostLoad = (unsigned int *) malloc(sizeof(unsigned int) * prop.numberOfTrialsBenchmark);
        unsigned int *deviceLoad;
        cudaMalloc(&deviceLoad, (sizeof(unsigned int) * prop.numberOfTrialsBenchmark));
        for (int initializeLoop = 0; initializeLoop < prop.numberOfTrialsBenchmark; initializeLoop++) {
            hostLoad[initializeLoop] = initializeLoop;
        }
        cudaMemcpy(deviceLoad, hostLoad, (sizeof(unsigned int) * prop.numberOfTrialsBenchmark), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        l1Benchmark<<<derivatives.numberOfBlocksPerMulp, info.warpSize>>>(devicePtr, laneLoop, deviceLoad, info, prop, derivatives);
        cudaDeviceSynchronize();
        cudaFree(deviceLoad);
        free(hostLoad);
    }
    cudaMemcpy(hostPtr, devicePtr, sizeof(data), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(devicePtr);
    for (int i = 0; i < derivatives.collectionSize; i++) {
        data.mulp[i] = (*hostPtr).mulp[i];
        data.warp[i] = (*hostPtr).warp[i];
        data.lane[i] = (*hostPtr).lane[i];
        data.time[i] = (*hostPtr).time[i];
    }
    cudaFreeHost(hostPtr);
    return data;
}

#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

//FINISHED

