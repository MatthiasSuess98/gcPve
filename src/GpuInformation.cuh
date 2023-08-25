#ifndef GCPVE_C_C_GPUINFORMATION_CUH
#define GCPVE_C_C_GPUINFORMATION_CUH

struct GpuInformation {
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
} GpuInformation;

GpuInformation getGpuInformation((int gpuId) {
    GpuInformation info;
    cudaDeviceProp deviceInfo{};
    cudaGetDeviceProperties(&deviceInfo, gpuId);
    info.name = deviceInfo.name;
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
    return info;
}

void createInfoFile(GpuInformation info) {
    char output[] = "GPU_Info.csv";
    FILE *csv = fopen(output, "w");
    fprintf(csv, "name; \"%s\"\n", info.name);
    fprintf(csv, "totalGlobalMem; \"%s\"\n", info.totalGlobalMem);
    fprintf(csv, "sharedMemPerBlock; \"%s\"\n", info.sharedMemPerBlock);
    fprintf(csv, "regsPerBlock; \"%s\"\n", info.regsPerBlock);
    fprintf(csv, "warpSize; \"%s\"\n", info.warpSize);
    fprintf(csv, "memPitch; \"%s\"\n", info.memPitch);
    fprintf(csv, "maxThreadsPerBlock; \"%s\"\n", info.maxThreadsPerBlock);
    fprintf(csv, "maxThreadsDim[3]; \"%s\"\n", info.maxThreadsDim[3]);
    fprintf(csv, "maxGridSize[3]; \"%s\"\n", info.maxGridSize[3]);
    fprintf(csv, "clockRate; \"%s\"\n", info.clockRate);
    fprintf(csv, "totalConstMem; \"%s\"\n", info.totalConstMem);
    fprintf(csv, "major; \"%s\"\n", info.major);
    fprintf(csv, "minor; \"%s\"\n", info.minor);
    fprintf(csv, "textureAlignment; \"%s\"\n", info.textureAlignment);
    fprintf(csv, "deviceOverlap; \"%s\"\n", info.deviceOverlap);
    fprintf(csv, "multiProcessorCount; \"%s\"\n", info.multiProcessorCount);
    fprintf(csv, "kernelExecTimeoutEnabled; \"%s\"\n", info.kernelExecTimeoutEnabled);
    fprintf(csv, "integrated; \"%s\"\n", info.integrated);
    fprintf(csv, "canMapHostMemory; \"%s\"\n", info.canMapHostMemory);
    fprintf(csv, "computeMode; \"%s\"\n", info.computeMode);
    fprintf(csv, "maxTexture1D; \"%s\"\n", info.maxTexture1D);
    fprintf(csv, "maxTexture2D[2]; \"%s\"\n", info.maxTexture2D[2]);
    fprintf(csv, "maxTexture3D[3]; \"%s\"\n", info.maxTexture3D[3]);
    fprintf(csv, "maxTexture1DLayered[2]; \"%s\"\n", info.maxTexture1DLayered[2]);
    fprintf(csv, "maxTexture2DLayered[3]; \"%s\"\n", info.maxTexture2DLayered[3]);
    fprintf(csv, "surfaceAlignment; \"%s\"\n", info.surfaceAlignment);
    fprintf(csv, "concurrentKernels; \"%s\"\n", info.concurrentKernels);
    fprintf(csv, "ECCEnabled; \"%s\"\n", info.ECCEnabled);
    fprintf(csv, "pciBusID; \"%s\"\n", info.pciBusID);
    fprintf(csv, "pciDeviceID; \"%s\"\n", info.pciDeviceID);
    fprintf(csv, "pciDomainID; \"%s\"\n", info.pciDomainID);
    fprintf(csv, "tccDriver; \"%s\"\n", info.tccDriver);
    fprintf(csv, "asyncEngineCount; \"%s\"\n", info.asyncEngineCount);
    fprintf(csv, "unifiedAddressing; \"%s\"\n", info.unifiedAddressing);
    fprintf(csv, "memoryClockRate; \"%s\"\n", info.memoryClockRate);
    fprintf(csv, "memoryBusWidth; \"%s\"\n", info.memoryBusWidth);
    fprintf(csv, "l2CacheSize; \"%s\"\n", info.l2CacheSize);
    fprintf(csv, "maxThreadsPerMultiProcessor; \"%s\"\n", info.maxThreadsPerMultiProcessor);
    fclose(csv);
}

#endif //GCPVE_C_C_GPUINFORMATION_CUH

