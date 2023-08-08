#include <cstdio>
#include <cuda.h>

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

int getCoreNumber(char* cmd) {
    printf("Execute command to get number of cores: %s\n", cmd);
#ifdef _WIN32
    if (strstr(cmd, "nvidia-settings") != nullptr) {
        printf("nvidia-settings does not work for windows\n");
        return 0;
    } else if(strstr(cmd, "deviceQuery.exe") == nullptr) {
        printf("It is required to use deviceQuery.exe\n");
        return 0;
    }
#else
    if (strstr(cmd, "nvidia-settings") != nullptr && strstr(cmd, "deviceQuery") != nullptr)  {
        printf("Nvidia-settings or deviceQuery not in command!\n");
        return 0;
    }
#endif
    FILE *p;
#ifdef _WIN32
    p = _popen(cmd, "r");
#else
    p = popen(cmd, "r");
#endif
    if (p == nullptr) {
        printf("Could not execute command %s!\n", cmd);
    }

    int totalNumOfCores;
    if (strstr(cmd, "deviceQuery") != nullptr) {
        printf("Using deviceQuery option for number of cores\n");
        char line[MAX_LINE_LENGTH] = {0};

        while (fgets(line, MAX_LINE_LENGTH, p)) {
            if (strstr(line, "core") || strstr(line, "Core")) {
                totalNumOfCores = parseCoreLine(line);
                break;
            }
        }
    } else {
        printf("Using nvidia-settings option for number of cores\n");
        char num[16] = {0};
        fgets(num, 16, p);
        totalNumOfCores = cvtCharArrToInt(num);
    }

#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    return totalNumOfCores;
}

CudaDeviceInfo getDeviceProperties(char* nviCoreCmd, int coreSwitch, int deviceID) {
    CudaDeviceInfo info;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp{};
    if (deviceID >= deviceCount) {
        deviceID = 0;
    }
    cudaGetDeviceProperties(&deviceProp, deviceID);
#ifdef _WIN32
    strcpy_s(info.GPUname, deviceProp.name);
#else
    strcpy(info.GPUname, deviceProp.name);
#endif
    info.cudaVersion = (float)deviceProp.major + (float)((float)deviceProp.minor / 10.);
    info.sharedMemPerThreadBlock = deviceProp.sharedMemPerBlock;
    info.sharedMemPerSM = deviceProp.sharedMemPerMultiprocessor;
    info.numberOfSMs = deviceProp.multiProcessorCount;
    info.registersPerThreadBlock = deviceProp.regsPerBlock;
    info.registersPerSM = deviceProp.regsPerMultiprocessor;
    info.cudaMaxGlobalMem = deviceProp.totalGlobalMem;
    info.cudaMaxConstMem = deviceProp.totalConstMem;
    info.L2CacheSize = deviceProp.l2CacheSize;
    info.memClockRate = deviceProp.memoryClockRate;
    info.memBusWidth = deviceProp.memoryBusWidth;
    info.GPUClockRate = deviceProp.clockRate;
    info.maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    if (coreSwitch == 0) {
        printf("Using helper_cuda option for number of cores\n");
        info.numberOfCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * info.numberOfSMs;
    } else {
        info.numberOfCores = getCoreNumber(nviCoreCmd);
    }
    return info;
}

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

