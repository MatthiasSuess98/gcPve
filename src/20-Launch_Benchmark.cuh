#ifndef GCPVE_20_L1_CACHE_LAUNCHER_CUH
#define GCPVE_20_L1_CACHE_LAUNCHER_CUH

#include <vector>

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

#include "21-L1_SM_L2_GM_Benchmark.cuh"


/**
 * Launches the benchmark kernel.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 * @param data The data collection in which the benchmark results will be written into.
 * @return The full new data collection.
 */
dataCollection launchBenchmarks(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives, dataCollection data) {

    // Perform the launch loops.
    for (int i = 0; i < info.multiProcessorCount; i++) {
        for (int j = 0; j < (derivatives.hardwareWarpsPerSm * prop.collectionFactor); j++) {

            // Allocate the memory for the benchmark.
            float *hostTime = nullptr;
            cudaMallocHost((void **) &hostTime, (sizeof(float) * info.warpSize * 4));
            unsigned int *hostLoad = nullptr;
            cudaMallocHost((void **) &hostLoad, (sizeof(unsigned int) * prop.numberOfTrialsBenchmark));
            float *deviceTime = nullptr;
            cudaMalloc((void **) &deviceTime, (sizeof(float) * info.warpSize * 4));
            unsigned int *deviceLoad = nullptr;
            cudaMalloc((void **) &deviceLoad, (sizeof(unsigned int) * prop.numberOfTrialsBenchmark));

            // Initialize the load.
            for (int k = 0; k < prop.numberOfTrialsBenchmark; k++) {
                hostLoad[k] = (k + 1) % prop.numberOfTrialsBenchmark;
            }

            // Copy the load, launch the benchmark and copy the time.
            cudaMemcpy((void *) deviceLoad, (void *) hostLoad, (sizeof(unsigned int) * prop.numberOfTrialsBenchmark), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            benchmark<<<(info.multiProcessorCount * derivatives.hardwareWarpsPerSm), info.warpSize>>>(info, prop, derivatives, deviceLoad, deviceTime, i, j);
            cudaDeviceSynchronize();
            cudaMemcpy((void *) hostTime, (void *) deviceTime, (sizeof(float) * info.warpSize * 4), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            // Copy the time data into the data collection.
            for (int k = 0; k < info.warpSize; k++) {
                data.mulp[(i * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize) + (j * info.warpSize) + k] = i;
                data.lane[(i * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize) + (j * info.warpSize) + k] = k;
                data.timeL1[(i * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize) + (j * info.warpSize) + k] = hostTime[k + (0 * info.warpSize)];
                data.timeSM[(i * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize) + (j * info.warpSize) + k] = hostTime[k + (1 * info.warpSize)];
                data.timeL2[(i * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize) + (j * info.warpSize) + k] = hostTime[k + (2 * info.warpSize)];
                data.timeGM[(i * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize) + (j * info.warpSize) + k] = hostTime[k + (3 * info.warpSize)];
            }

            // Free the allocated memory.
            cudaFreeHost(hostTime);
            cudaFreeHost(hostLoad);
            cudaFree(deviceTime);
            cudaFree(deviceLoad);

            // Signal that the benchmark is still running.
            printf(".");
        }

        // Signal that the benchmark is still running.
        printf("\n");
    }

    // Return the final data collection.
    return data;
}


#endif //GCPVE_20_L1_CACHE_LAUNCHER_CUH

//FINISHED

