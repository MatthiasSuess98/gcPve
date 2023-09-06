#include <cstdio>
#include <cuda.h>

#include "1_Gpu_Information.cuh"
#include "2_Analyze_SM.cuh"

void createBenchmarks(int gpuId) {
    GpuInformation gpuInfo = getGpuInformation(gpuId);
    createInfoFile(gpuInfo);
    SmSimpleAddBenchmark16bit smSimpleAddBenchmark;
    int counter;
    int averageTime;
    int averageLane;
    int averageWarp;
    int averageSm;
    char output1[] = "Benchmark_16bit.csv";
    FILE *csv1 = fopen(output1, "w");
    for (int i = 0; i < gpuInfo.multiProcessorCount; i++) {
        smSimpleAddBenchmark = analyzeSm(i, gpuInfo);
        averageTime = 0;
        averageLane = 0;
        averageWarp = 0;
        averageSm = 0;
        counter = 0;
        for (int j = 0; j < 65536; j++) {
            if (smSimpleAddBenchmark[i].correctSm[j]) {
                averageTime = averageTime + smSimpleAddBenchmark[i].finalTime;
                averageLane = averageLane + smSimpleAddBenchmark[i].laneFinal;
                averageWarp = averageWarp + smSimpleAddBenchmark[i].warpFinal;
                averageSm = averageSm + smSimpleAddBenchmark[i].smFinal;
                counter++;
            }
        }
        fprintf(csv1, "%d ; ", i);
        fprintf(csv1, "%f ; ", (((float) averageLane) / ((float) counter)));
        fprintf(csv1, "%f ; ", (((float) averageWarp) / ((float) counter)));
        fprintf(csv1, "%f ; ", (((float) averageSm) / ((float) counter)));
        fprintf(csv1, "%f ; ", (((float) averageTime) / ((float) counter)));
    }
    fclose(csv1);
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
        }
    } else {
        printf("Please select the GPU for which the benchmarks should be created.\n");
        printf("To do so, use the following syntax (here for GPU 0): \"gcPve 0\"");
        printf("To get a list of all available GPUs use the command \"nvidia-smi -L\".\n");
        printf("It is also possible to select multiple GPUs by appending multiple numbers.\n");
    }
}

