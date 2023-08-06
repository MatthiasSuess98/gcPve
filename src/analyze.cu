#include <cstdio>
#include <climits>

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
    int coreQuerySize 1024;
    char cudaCoreQueryPath[coreQuerySize];
    graphicCardInformation cardInformation = getDeviceProperties(cudaCoreQueryPath, coreSwitch, deviceID);
    createOutputFile(cardInformation);
}

void createOutputFile(graphicCardInformation cardInformation) {



    printf("\n\n**************************************************\n");
    printf("\tPRINT GPU BENCHMARK RESULT\n");
    printf("**************************************************\n\n");

    char outputCSV[] = "GPU_Memory_Topology.csv";
    FILE *csv = fopen(outputCSV, "w");
    if (csv == nullptr) {
        printf("[WARNING]: Cannot open file for writing - close csv file if currently open\n");
        csv = stdout;
    }

    printf("GPU name: %s\n\n", cudaInfo.GPUname);
    fprintf(csv, "GPU_INFORMATION; GPU_vendor; \"Nvidia\"; GPU_name; \"%s\"\n", cudaInfo.GPUname);

    printf("PRINT COMPUTE RESOURCE INFORMATION:\n");
    fprintf(csv, "COMPUTE_RESOURCE_INFORMATION; ");
    printf("CUDA compute capability: %.2f\n", cudaInfo.cudaVersion);
    fprintf(csv, "CUDA_compute_capability; \"%.2f\"; ", cudaInfo.cudaVersion);
    printf("Number Of streaming multiprocessors: %d\n", cudaInfo.numberOfSMs);
    fprintf(csv, "Number_of_streaming_multiprocessors; %d; ", cudaInfo.numberOfSMs);
    printf("Number Of Cores in GPU: %d\n", cudaInfo.numberOfCores);
    fprintf(csv, "Number_of_cores_in_GPU; %d; ", cudaInfo.numberOfCores);
    printf("Number Of Cores/SM in GPU: %d\n\n", cudaInfo.numberOfCores / cudaInfo.numberOfSMs);
    fprintf(csv, "Number_of_cores_per_SM; %d\n", cudaInfo.numberOfCores / cudaInfo.numberOfSMs);

    printf("PRINT REGISTER INFORMATION:\n");
    fprintf(csv, "REGISTER_INFORMATION; ");
    printf("Registers per thread block: %d 32-bit registers\n", cudaInfo.registersPerThreadBlock);
    fprintf(csv, "Registers_per_thread_block; %d; \"32-bit registers\"; ", cudaInfo.registersPerThreadBlock);
    printf("Registers per SM: %d 32-bit registers\n\n", cudaInfo.registersPerSM);
    fprintf(csv, "Registers_per_SM; %d; \"32-bit registers\"\n", cudaInfo.registersPerSM);

    printf("PRINT ADDITIONAL INFORMATION:\n");
    fprintf(csv, "ADDITIONAL_INFORMATION; ");

    double val;
    unsigned int originalFrequency = cudaInfo.memClockRate;
    const char* MemClockFreqUnit = getSizeNiceFormatHertz(&val, originalFrequency);
    printf("Memory Clock Frequency: %.3f %s\n", val, MemClockFreqUnit);
    fprintf(csv, "Memory_Clock_Frequency; %.3f; \"%s\"; ", val, MemClockFreqUnit);
    printf("Memory Bus Width: %d bits\n", cudaInfo.memBusWidth);
    fprintf(csv, "Memory_Bus_Width; %d; \"bit\"; ", cudaInfo.memBusWidth);

    originalFrequency = cudaInfo.GPUClockRate;
    const char* GPUClockFreqUnit = getSizeNiceFormatHertz(&val, originalFrequency);
    printf("GPU Clock rate: %.3f %s\n\n", val, GPUClockFreqUnit);
    fprintf(csv, "GPU_Clock_Rate; %.3f; \"%s\"\n", val, GPUClockFreqUnit);

    fprintf(csv, "L1_DATA_CACHE; ");
    if (!L1_global_load_enabled) {
        printf("L1 DATA CACHE INFORMATION missing: GPU does not allow caching of global loads in L1\n");
        fprintf(csv, "\"N/A\"\n");
    } else {
        if (result[L1].benchmarked) {
            printf("PRINT L1 DATA CACHE INFORMATION:\n");

            if (result[L1].CacheSize.realCP) {
                double size;
                size_t original = result[L1].CacheSize.CacheSize;
                const char* unit = getSizeNiceFormatByte(&size, original);
                printf("Detected L1 Data Cache Size: %f %s\n", size, unit);
                fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
            } else {
                double size;
                size_t original = result[L1].CacheSize.maxSizeBenchmarked;
                const char* unit = getSizeNiceFormatByte(&size, original);
                printf("Detected L1 Data Cache Size: >= %f %s\n", size, unit);
                fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
            }
            printf("Detected L1 Data Cache Line Size: %d B\n", result[L1].cacheLineSize);
            fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[L1].cacheLineSize);
            printf("Detected L1 Data Cache Load Latency: %d cycles\n", result[L1].latencyCycles);
            fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[L1].latencyCycles);
            printf("Detected L1 Data Cache Load Latency: %d nanoseconds\n", result[L1].latencyNano);
            fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[L1].latencyNano);
            printf("L1 Data Cache Is Shared On %s-level\n", shared_where[L1]);
            fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[L1]);
            printf("Does L1 Data Cache Share the physical cache with the Texture Cache? %s\n", L1ShareTexture ? "Yes" : "No");
            fprintf(csv, "Share_Cache_With_Texture; %d; ", L1ShareTexture);
            printf("Does L1 Data Cache Share the physical cache with the Read-Only Cache? %s\n", ROShareL1Data ? "Yes" : "No");
            fprintf(csv, "Share_Cache_With_Read-Only; %d; ", ROShareL1Data);
            printf("Does L1 Data Cache Share the physical cache with the Constant L1 Cache? %s\n", L1ShareConst ? "Yes" : "No");
            fprintf(csv, "Share_Cache_With_ConstantL1; %d; ", L1ShareConst);
            //printf("Detected L1 Cache Load Bandwidth: %llu MB / s\n\n", result[L1].bandwidth);
            printf("Detected L1 Data Caches Per SM: %d\n\n", result[L1].numberPerSM);
            fprintf(csv, "Caches_Per_SM; %d\n", result[L1].numberPerSM);
        } else {
            printf("L1 Data CACHE WAS NOT BENCHMARKED!\n\n");
            fprintf(csv, "\"N/A\"\n");
        }
    }

    fprintf(csv, "L2_DATA_CACHE; ");
    if (result[L2].benchmarked) {
        printf("PRINT L2 CACHE INFORMATION:\n");
        if (result[L2].CacheSize.realCP) {
            double size;
            size_t original = result[L2].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected L2 Cache Size: %.3f %s\n", size, unit);
            fprintf(csv, "Size; %.3f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[L2].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected L2 Cache Size: >= %.3f %s\n", size, unit);
            fprintf(csv, "Size; %.3f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected L2 Cache Line Size: %d B\n", result[L2].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[L2].cacheLineSize);
        printf("Detected L2 Cache Load Latency: %d cycles\n", result[L2].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[L2].latencyCycles);
        printf("Detected L2 Cache Load Latency: %d nanoseconds\n", result[L2].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[L2].latencyNano);
        printf("L2 Cache Is Shared On %s-level\n", shared_where[L2]);
        fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[L2]);
        printf("Detected L2 Data Cache Segments Per GPU: %d\n\n", result[L2].numberPerSM);
        fprintf(csv, "Caches_Per_GPU; %d\n", result[L2].numberPerSM);
    } else {
        printf("L2 CACHE WAS NOT BENCHMARKED!\n\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "TEXTURE_CACHE; ");
    if (result[Texture].benchmarked) {
        printf("PRINT TEXTURE CACHE INFORMATION:\n");
        if (result[Texture].CacheSize.realCP) {
            double size;
            size_t original = result[Texture].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Texture Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[Texture].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Texture Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Texture Cache Line Size: %d B\n", result[Texture].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[Texture].cacheLineSize);
        printf("Detected Texture Cache Load Latency: %d cycles\n", result[Texture].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[Texture].latencyCycles);
        printf("Detected Texture Cache Load Latency: %d nanoseconds\n", result[Texture].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[Texture].latencyNano);
        printf("Texture Cache Is Shared On %s-level\n", shared_where[Texture]);
        fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[Texture]);
        printf("Does Texture Cache Share the physical cache with the L1 Data Cache? %s\n", L1ShareTexture ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_L1_Data; %d; ", L1ShareTexture);
        printf("Does Texture Cache Share the physical cache with the Read-Only Cache? %s\n", ROShareTexture ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_Read-Only; %d; ", ROShareTexture);
        printf("Detected Texture Caches Per SM: %d\n\n", result[Texture].numberPerSM);
        fprintf(csv, "Caches_Per_SM; %d\n", result[Texture].numberPerSM);
    } else {
        printf("TEXTURE CACHE WAS NOT BENCHMARKED!\n\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "READ-ONLY_CACHE; ");
    if (result[RO].benchmarked) {
        printf("PRINT Read-Only CACHE INFORMATION:\n");
        if (result[RO].CacheSize.realCP) {
            double size;
            size_t original = result[RO].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Read-Only Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[RO].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Read-Only Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Read-Only Cache Line Size: %d B\n", result[RO].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[RO].cacheLineSize);
        printf("Detected Read-Only Cache Load Latency: %d cycles\n", result[RO].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[RO].latencyCycles);
        printf("Detected Read-Only Cache Load Latency: %d nanoseconds\n", result[RO].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[RO].latencyNano);
        printf("Read-Only Cache Is Shared On %s-level\n", shared_where[RO]);
        fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[RO]);
        printf("Does Read-Only Cache Share the physical cache with the L1 Data Cache? %s\n", ROShareL1Data ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_L1_Data; %d; ", ROShareL1Data);
        printf("Does Read-Only Cache Share the physical cache with the Texture Cache? %s\n", ROShareTexture ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_Texture; %d; ", ROShareTexture);
        printf("Detected Read-Only Caches Per SM: %d\n\n", result[RO].numberPerSM);
        fprintf(csv, "Caches_Per_SM; %d\n", result[RO].numberPerSM);
    } else {
        printf("READ-ONLY CACHE WAS NOT BENCHMARKED!\n\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "CONSTANT_L1_CACHE; ");
    if (result[Const1].benchmarked) {
        printf("PRINT CONSTANT CACHE L1 INFORMATION:\n");
        if (result[Const1].CacheSize.realCP) {
            double size;
            size_t original = result[Const1].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1 Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[Const1].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1 Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Constant L1 Cache Line Size: %d B\n", result[Const1].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[Const1].cacheLineSize);
        printf("Detected Constant L1 Cache Load Latency: %d cycles\n", result[Const1].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[Const1].latencyCycles);
        printf("Detected Constant L1 Cache Load Latency: %d nanoseconds\n", result[Const1].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[Const1].latencyNano);
        printf("Constant L1 Cache Is Shared On %s-level\n", shared_where[Const1]);
        fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[Const1]);
        printf("Does Constant L1 Cache Share the physical cache with the L1 Data Cache? %s\n\n", L1ShareConst ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_L1_Data; %d; ", L1ShareConst);
        printf("Detected Constant L1 Caches Per SM: %d\n\n", result[Const1].numberPerSM);
        fprintf(csv, "Caches_Per_SM; %d\n", result[Const1].numberPerSM);
    } else {
        printf("CONSTANT CACHE L1 WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "CONST_L1_5_CACHE; ");
    if (result[Const2].benchmarked) {
        printf("PRINT CONSTANT L1.5 CACHE INFORMATION:\n");
        if (result[Const2].CacheSize.realCP) {
            double size;
            size_t original = result[Const2].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1.5 Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[Const2].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1.5 Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Constant L1.5 Cache Line Size: %d B\n", result[Const2].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[Const2].cacheLineSize);
        printf("Detected Constant L1.5 Cache Load Latency: %d cycles\n", result[Const2].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[Const2].latencyCycles);
        printf("Detected Constant L1.5 Cache Load Latency: %d nanoseconds\n", result[Const2].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[Const2].latencyNano);
        printf("Const L1.5 Cache Is Shared On %s-level\n\n", shared_where[Const2]);
        fprintf(csv, "Shared_On; \"%s-level\"\n", shared_where[Const2]);
    } else {
        printf("CONSTANT CACHE L1.5 WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "MAIN_MEMORY; ");
    if (result[MAIN].benchmarked) {
        printf("PRINT MAIN MEMORY INFORMATION:\n");
        if (result[MAIN].CacheSize.realCP) {
            double size;
            size_t original = result[MAIN].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Main Memory Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[MAIN].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Main Memory Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Main Memory Load Latency: %d cycles\n", result[MAIN].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[MAIN].latencyCycles);
        printf("Detected Main Memory Load Latency: %d nanoseconds\n", result[MAIN].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[MAIN].latencyNano);
        printf("Main Memory Is Shared On %s-level\n\n", shared_where[MAIN]);
        fprintf(csv, "Shared_On; \"%s-level\"\n", shared_where[MAIN]);
    } else {
        printf("MAIN MEMORY WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "SHARED_MEMORY; ");
    if (result[SHARED].benchmarked) {
        printf("PRINT SHARED MEMORY INFORMATION:\n");
        if (result[SHARED].CacheSize.realCP) {
            double size;
            size_t original = result[SHARED].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Shared Memory Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[SHARED].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Shared Memory Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Shared Memory Load Latency: %d cycles\n", result[SHARED].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[SHARED].latencyCycles);
        printf("Detected Shared Memory Load Latency: %d nanoseconds\n", result[SHARED].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[SHARED].latencyNano);
        printf("Shared Memory Is Shared On %s-level\n\n", shared_where[SHARED]);
        fprintf(csv, "Shared_On; \"%s-level\"\n", shared_where[SHARED]);
    } else {
        printf("SHARED MEMORY WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }
    fclose(csv);
}
