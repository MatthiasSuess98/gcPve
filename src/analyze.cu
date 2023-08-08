#include <cstdio>

typedef struct CudaDeviceInfo {
    char GPUname[256];
    float cudaVersion;
    int numberOfSMs;
    int numberOfCores;
    size_t sharedMemPerThreadBlock;
    size_t sharedMemPerSM;
    int registersPerThreadBlock;
    int registersPerSM;
    size_t cudaMaxGlobalMem;
    size_t  cudaMaxConstMem;
    int L2CacheSize;
    int memClockRate;
    int memBusWidth;
    int GPUClockRate;
    int maxThreadsPerBlock;
} CudaDeviceInfo;

CudaDeviceInfo getDeviceProperties(char* nviCoreCmd, int coreSwitch, int deviceID) {
    CudaDeviceInfo info;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp{};
    if (deviceID >= deviceCount) {
        deviceID = 0;
    }
    cudaGetDeviceProperties(&deviceProp, deviceID);

void createOutputFile(CudaDeviceInfo cardInformation) {
    printf("Create the output file...\n");
    char output[] = "Output";
    FILE *csv = fopen(output, "w");
    if (csv == nullptr) {
        printf("[WARNING]: Cannot open output file for writing.\n");
        csv = stdout;
    }
    fprintf(csv, "GPU_vendor; \"%s\"; ", "Nvidia");
    fprintf(csv, "GPU_name; \"%s\"; ", cardInformation.GPUname);
    fprintf(csv, "CUDA_compute_capability; \"%.2f\"; ", cardInformation.cudaVersion);
    fprintf(csv, "Number_of_streaming_multiprocessors; %d; ", cardInformation.numberOfSMs);
    fprintf(csv, "Number_of_cores_in_GPU; %d; ", cardInformation.numberOfCores);
    fprintf(csv, "Number_of_cores_per_SM; %d; ", cardInformation.numberOfCores / cardInformation.numberOfSMs);
    fprintf(csv, "Registers_per_thread_block; %d; \"32-bit registers\"; ", cardInformation.registersPerThreadBlock);
    fprintf(csv, "Registers_per_SM; %d; \"32-bit registers\"; ", cardInformation.registersPerSM);
    fclose(csv);
}

int main(int argCount, char *argVariables[]) {
    for (int i = 1; i < argCount; i++) {
        char *arg = argVariables[i];
        if (strcmp(arg, "-adv") == 0) {
            printf("The final file will provide advanced GPU information.\n");
        } else if (strcmp(arg, "-fas") == 0) {
            printf("The benchmarks will be executed faster than normal.\n");
        } else {
            printf("No commands were given.\n");
        }
    }
    int coreSwitch = 0;
    int deviceID = 0;
    int coreQuerySize = 1024;
    char cudaCoreQueryPath[coreQuerySize];
    CudaDeviceInfo cardInformation = getDeviceProperties(cudaCoreQueryPath, coreSwitch, deviceID);
    createOutputFile(cardInformation);
}

