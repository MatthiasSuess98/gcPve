#ifndef GCPVE_00_MAIN_CU
#define GCPVE_00_MAIN_CU

#include <cstdio>
#include <cuda.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace fs = std::filesystem;

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"
#include "10-Perform_Benchmark.cuh"
#include "21-L1_Cache_Benchmark.cuh"
#include "31-SM_Cache_Benchmark.cuh"
#include "41-L2_Cache_Benchmark.cuh"
#include "51-GM_Cache_Benchmark.cuh"

/**
 * Create all Benchmarks for the selected GPU.
 * @param gpuId The selected GPU.
 */
void createBenchmarks(int gpuId) {

    // Create the necessary directories.
    fs::create_directory("raw");
    fs::create_directory("out");

    // Determine all information, properties and derivatives for the selected GPU.
    GpuInformation info = getGpuInformation(gpuId);
    createInfoFile(info);
    BenchmarkProperties prop = getBenchmarkProperties();
    createPropFile(prop);
    InfoPropDerivatives derivatives = getInfoPropDerivatives(info, prop);
    createInfoPropDerivatives(derivatives);

    // Perform the benchmarks.
    launchL1Benchmark(info, prop, derivatives);
    launchSMBenchmark(info, prop, derivatives);
    launchL2Benchmark(info, prop, derivatives);
    launchGMBenchmark(info, prop, derivatives);

    // Call the python file.
    FILE *p;
    p = popen("python3 evaluation.py", "r");
    pclose(p);
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
                    printf("[INFO] The benchmark started.\n");
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

