#ifndef GCPVE_00_MAIN_CU
#define GCPVE_00_MAIN_CU

#include <cstdio>
#include <cuda.h>

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"
#include "10-Perform_Benchmark.cuh"

/**
 * Create all Benchmarks for the selected GPU.
 * @param gpuId The selected GPU.
 */
void createBenchmarks(int gpuId) {

    // Determine all information, properties and derivatives for the selected GPU.
    GpuInformation info = getGpuInformation(gpuId);
    createInfoFile(info);
    BenchmarkProperties prop = getBenchmarkProperties();
    createPropFile(prop);
    InfoPropDerivatives derivatives = getInfoPropDerivatives(info, prop);
    createInfoPropDerivatives(derivatives);

    // Perform the benchmarks.
    performSmallBenchmark(info, prop, derivatives);
}


/**
 * Main function of the program which is initially called.
 * @param argCount Number of given parameters.
 * @param argVariables The given parameters.
 * @return For stopping the program it returns the value zero.
 */
int main(int argCount, char *argVariables[]) {

    // Interpretation of the given parameters.
    // argVariables[0] is the command.
    if (argCount >= 2) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
            for (int i = 1; i < argCount; i++) {
                char *ptr;
                int gpuId = strtol(argVariables[i], &ptr, 10);
                if (*ptr || (gpuId >= deviceCount)) {
                    printf("[ERROR] There is no GPU \"%s\".\n", argVariables[i]);
                } else {
                    printf("[INFO] The Benchmark started.");
                    createBenchmarks(gpuId);
                }
            }
    } else {
        printf("[ERROR] Please select the GPU for which the benchmarks should be created.\n");
    }

    // Stopping the program.
    return 0;
}

#endif //GCPVE_00_MAIN_CU

//FINISHED

