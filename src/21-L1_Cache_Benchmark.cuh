#ifndef GCPVE_21_L1_CACHE_BENCHMARK_CUH
#define GCPVE_21_L1_CACHE_BENCHMARK_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

/**
 *
 */
__global__ void smallL1Benchmark(unsigned int *deviceLoad, float *deviceTime, int i) {

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

        for (int j = 0; j < 1024; j++) {
            ptr = deviceLoad + value;
            asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
        }

        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));

        for (int j = 0; j < 1024; j++) {
            ptr = deviceLoad + value;
            asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
        }

        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));

        unsigned int saveValue = value;
        saveValue++;

        deviceTime[lane] = ((float) (endTime - startTime)) / 1024;
        printf("%lld ", (endTime - startTime));

    }
}


/**
 *
 */
void launchL1Benchmark(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    char output[] = "raw/Benchmark_L1.csv";
    FILE *csv = fopen(output, "w");
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 4; j++) {

            float *hostTime = nullptr;
            cudaMallocHost((void **) &hostTime, (sizeof(float) * 32));
            unsigned int *hostLoad = nullptr;
            cudaMallocHost((void **) &hostLoad, (sizeof(unsigned int) * 1024));
            float *deviceTime = nullptr;
            cudaMalloc((void **) &deviceTime, (sizeof(float) * 32));
            unsigned int *deviceLoad = nullptr;
            cudaMalloc((void **) &deviceLoad, (sizeof(unsigned int) * 1024));

            for (int k = 0; k < 1024; k++) {
                hostLoad[k] = (k * 512) % 1024;
            }

            cudaMemcpy((void *) deviceLoad, (void *) hostLoad, (sizeof(unsigned int) * 1024), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            smallL1Benchmark<<<(4 * 30), (4 * 32)>>>(deviceLoad, deviceTime, i);
            cudaDeviceSynchronize();

            cudaMemcpy((void *) hostTime, (void *) deviceTime, (sizeof(float) * 32), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            for (int k = 0; k < 32; k++) {
                fprintf(csv, "%f", hostTime[k]);
                if (k < (32 - 1)) {
                    fprintf(csv, ";");
                }
            }

            cudaFreeHost(hostTime);
            cudaFreeHost(hostLoad);
            cudaFree(deviceTime);
            cudaFree(deviceLoad);

            fprintf(csv, "\n");
        }
        fprintf(csv, "\n");
    }
    fclose(csv);
}

#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

//FINISHED

