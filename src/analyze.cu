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
    printf("Create the output file...\n");
    char output[] = "Output";
    FILE *csv = fopen(output, "w");
    if (csv == nullptr) {
        printf("[WARNING]: Cannot open output file for writing.\n");
        csv = stdout;
    }
    fprintf(csv, "GPU_vendor; \"%s\"; ", "Nvidia");
    fprintf(csv, "GPU_name; \"%s\"; ", cardInformation.GPUname);
    fprintf(csv, "CUDA_compute_capability; \"%.2f\"; ", cudaInfo.cudaVersion);
    fprintf(csv, "Number_of_streaming_multiprocessors; %d; ", cudaInfo.numberOfSMs);
    fprintf(csv, "Number_of_cores_in_GPU; %d; ", cudaInfo.numberOfCores);
    fprintf(csv, "Number_of_cores_per_SM; %d; ", cudaInfo.numberOfCores / cudaInfo.numberOfSMs);
    fprintf(csv, "Registers_per_thread_block; %d; \"32-bit registers\"; ", cudaInfo.registersPerThreadBlock);
    fprintf(csv, "Registers_per_SM; %d; \"32-bit registers\"; ", cudaInfo.registersPerSM);
    fclose(csv);
}

