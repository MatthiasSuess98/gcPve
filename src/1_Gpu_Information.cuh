typedef struct GpuInformation {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture2D[2];
    int maxTexture3D[3];
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    float cudaVersion;
    int numberOfCores;
    int maxNumberOfWarpsPerSm;
} GpuInformation;

int getNumberOfCores(int major, int minor) {
    // Data from "helper_cuda.h".
    if ((major == 3) && (minor == 0)) {
        return 192;
    } else if ((major == 3) && (minor == 2)) {
        return 192;
    } else if ((major == 3) && (minor == 5)) {
        return 192;
    } else if ((major == 3) && (minor == 7)) {
        return 192;
    } else if ((major == 5) && (minor == 0)) {
        return 128;
    } else if ((major == 5) && (minor == 2)) {
        return 128;
    } else if ((major == 5) && (minor == 3)) {
        return 128;
    } else if ((major == 6) && (minor == 0)) {
        return 64;
    } else if ((major == 6) && (minor == 1)) {
        return 128;
    } else if ((major == 6) && (minor == 2)) {
        return 128;
    } else if ((major == 7) && (minor == 0)) {
        return 64;
    } else if ((major == 7) && (minor == 2)) {
        return 64;
    } else if ((major == 7) && (minor == 5)) {
        return 64;
    } else if ((major == 8) && (minor == 0)) {
        return 64;
    } else if ((major == 8) && (minor == 6)) {
        return 128;
    } else if ((major == 8) && (minor == 7)) {
        return 128;
    } else if ((major == 8) && (minor == 9)) {
        return 128;
    } else if ((major == 9) && (minor == 0)) {
        return 128;
    } else {
        return 0;
    }
}

GpuInformation getGpuInformation(int gpuId) {
    GpuInformation info;
    cudaDeviceProp deviceInfo{};
    cudaGetDeviceProperties(&deviceInfo, gpuId);
    strcpy(info.name, deviceInfo.name);
    info.totalGlobalMem = deviceInfo.totalGlobalMem;
    info.sharedMemPerBlock = deviceInfo.sharedMemPerBlock;
    info.regsPerBlock = deviceInfo.regsPerBlock;
    info.warpSize = deviceInfo.warpSize;
    info.memPitch = deviceInfo.memPitch;
    info.maxThreadsPerBlock = deviceInfo.maxThreadsPerBlock;
    info.maxThreadsDim[3] = deviceInfo.maxThreadsDim[3];
    info.maxGridSize[3] = deviceInfo.maxGridSize[3];
    info.clockRate = deviceInfo.clockRate;
    info.totalConstMem = deviceInfo.totalConstMem;
    info.major = deviceInfo.major;
    info.minor = deviceInfo.minor;
    info.textureAlignment = deviceInfo.textureAlignment;
    info.deviceOverlap = deviceInfo.deviceOverlap;
    info.multiProcessorCount = deviceInfo.multiProcessorCount;
    info.kernelExecTimeoutEnabled = deviceInfo.kernelExecTimeoutEnabled;
    info.integrated = deviceInfo.integrated;
    info.canMapHostMemory = deviceInfo.canMapHostMemory;
    info.computeMode = deviceInfo.computeMode;
    info.maxTexture1D = deviceInfo.maxTexture1D;
    info.maxTexture2D[2] = deviceInfo.maxTexture2D[2];
    info.maxTexture3D[3] = deviceInfo.maxTexture3D[3];
    info.maxTexture1DLayered[2] = deviceInfo.maxTexture1DLayered[2];
    info.maxTexture2DLayered[3] = deviceInfo.maxTexture2DLayered[3];
    info.surfaceAlignment = deviceInfo.surfaceAlignment;
    info.concurrentKernels = deviceInfo.concurrentKernels;
    info.ECCEnabled = deviceInfo.ECCEnabled;
    info.pciBusID = deviceInfo.pciBusID;
    info.pciDeviceID = deviceInfo.pciDeviceID;
    info.pciDomainID = deviceInfo.pciDomainID;
    info.tccDriver = deviceInfo.tccDriver;
    info.asyncEngineCount = deviceInfo.asyncEngineCount;
    info.unifiedAddressing = deviceInfo.unifiedAddressing;
    info.memoryClockRate = deviceInfo.memoryClockRate;
    info.memoryBusWidth = deviceInfo.memoryBusWidth;
    info.l2CacheSize = deviceInfo.l2CacheSize;
    info.maxThreadsPerMultiProcessor = deviceInfo.maxThreadsPerMultiProcessor;
    info.cudaVersion = (((float) info.major) + (((float) info.minor) / 10.));
    info.numberOfCores = getNumberOfCores(info.major, info.minor);
    info.maxNumberOfWarpsPerSm = info.maxThreadsPerMultiProcessor / info.warpSize;
    return info;
}

void createInfoFile(GpuInformation info) {
    char output[] = "GPU_Info.csv";
    FILE *csv = fopen(output, "w");
    fprintf(csv, "name; \"%s\"\n", info.name);
    fprintf(csv, "totalGlobalMem; \"%zu\"\n", info.totalGlobalMem);
    fprintf(csv, "sharedMemPerBlock; \"%zu\"\n", info.sharedMemPerBlock);
    fprintf(csv, "regsPerBlock; \"%d\"\n", info.regsPerBlock);
    fprintf(csv, "warpSize; \"%d\"\n", info.warpSize);
    fprintf(csv, "memPitch; \"%zu\"\n", info.memPitch);
    fprintf(csv, "maxThreadsPerBlock; \"%d\"\n", info.maxThreadsPerBlock);
    fprintf(csv, "maxThreadsDim[3]; \"%d\"\n", info.maxThreadsDim[3]);
    fprintf(csv, "maxGridSize[3]; \"%d\"\n", info.maxGridSize[3]);
    fprintf(csv, "clockRate; \"%d\"\n", info.clockRate);
    fprintf(csv, "totalConstMem; \"%zu\"\n", info.totalConstMem);
    fprintf(csv, "major; \"%d\"\n", info.major);
    fprintf(csv, "minor; \"%d\"\n", info.minor);
    fprintf(csv, "textureAlignment; \"%zu\"\n", info.textureAlignment);
    fprintf(csv, "deviceOverlap; \"%d\"\n", info.deviceOverlap);
    fprintf(csv, "multiProcessorCount; \"%d\"\n", info.multiProcessorCount);
    fprintf(csv, "kernelExecTimeoutEnabled; \"%d\"\n", info.kernelExecTimeoutEnabled);
    fprintf(csv, "integrated; \"%d\"\n", info.integrated);
    fprintf(csv, "canMapHostMemory; \"%d\"\n", info.canMapHostMemory);
    fprintf(csv, "computeMode; \"%d\"\n", info.computeMode);
    fprintf(csv, "maxTexture1D; \"%d\"\n", info.maxTexture1D);
    fprintf(csv, "maxTexture2D[2]; \"%d\"\n", info.maxTexture2D[2]);
    fprintf(csv, "maxTexture3D[3]; \"%d\"\n", info.maxTexture3D[3]);
    fprintf(csv, "maxTexture1DLayered[2]; \"%d\"\n", info.maxTexture1DLayered[2]);
    fprintf(csv, "maxTexture2DLayered[3]; \"%d\"\n", info.maxTexture2DLayered[3]);
    fprintf(csv, "surfaceAlignment; \"%zu\"\n", info.surfaceAlignment);
    fprintf(csv, "concurrentKernels; \"%d\"\n", info.concurrentKernels);
    fprintf(csv, "ECCEnabled; \"%d\"\n", info.ECCEnabled);
    fprintf(csv, "pciBusID; \"%d\"\n", info.pciBusID);
    fprintf(csv, "pciDeviceID; \"%d\"\n", info.pciDeviceID);
    fprintf(csv, "pciDomainID; \"%d\"\n", info.pciDomainID);
    fprintf(csv, "tccDriver; \"%d\"\n", info.tccDriver);
    fprintf(csv, "asyncEngineCount; \"%d\"\n", info.asyncEngineCount);
    fprintf(csv, "unifiedAddressing; \"%d\"\n", info.unifiedAddressing);
    fprintf(csv, "memoryClockRate; \"%d\"\n", info.memoryClockRate);
    fprintf(csv, "memoryBusWidth; \"%d\"\n", info.memoryBusWidth);
    fprintf(csv, "l2CacheSize; \"%d\"\n", info.l2CacheSize);
    fprintf(csv, "maxThreadsPerMultiProcessor; \"%d\"\n", info.maxThreadsPerMultiProcessor);
    fprintf(csv, "cudaVersion; \"%f\"\n", info.cudaVersion);
    fprintf(csv, "numberOfCores; \"%d\"\n", info.numberOfCores);
    fprintf(csv, "maxNumberOfWarpsPerSm; \"%d\"\n", info.maxNumberOfWarpsPerSm);
    fclose(csv);
}

