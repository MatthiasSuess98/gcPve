#include <cstdio>
#include <cuda.h>

#include "01-Gpu_Information.cuh"
#include "10-Perform_Benchmark.cuh"

void createBenchmarks(int gpuId) {
    GpuInformation gpuInfo = getGpuInformation(gpuId);
    createInfoFile(gpuInfo);
    if (areThereWarpDifferences(gpuId, gpuInfo)) {
        //analyzeScheduling
    } else {
        //useWarpOne
    }
}

int main(int argCount, char *argVariables[]) {
    // argVariables[0] is the command.
    if (argCount >= 2) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
            for (int i = 1; i < argCount; i++) {
                char *ptr;
                int gpuId = strtol(argVariables[i], &ptr, 10);
                if (*ptr || (gpuId >= deviceCount)) {
                    printf("There is no GPU \"%d\".\n", gpuId);
                } else {
                    createBenchmarks(gpuId);
                }
            }
    } else {
        printf("Please select the GPU for which the benchmarks should be created.\n");
        printf("To do so, use the following syntax (here for GPU 0): \"gcPve 0\"");
        printf("To get a list of all available GPUs use the command \"nvidia-smi -L\".\n");
        printf("It is also possible to select multiple GPUs by appending multiple numbers.\n");
    }
}

