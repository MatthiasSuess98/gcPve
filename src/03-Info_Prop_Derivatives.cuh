#ifndef GCPVE_03_INFO_PROP_DERIVATIVES_CUH
#define GCPVE_03_INFO_PROP_DERIVATIVES_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"


/**
 * Data structure for all derivatives.
 */
typedef struct InfoPropDerivatives {

    // Full cuda version.
    float cudaVersion;

    // Basic derivatives.
    int numberOfCoresPerSm;
    int totalNumberOfCores;
    int hardwareWarpsPerSm;
    int maxNumberOfWarpsPerSm;

} InfoPropDerivatives;


/**
 * Creates a data structure with all derivatives.
 * @param info Information of the current GPU.
 * @param prop Properties of the current benchmarks.
 * @return Data structure with all derivatives.
 */
InfoPropDerivatives getInfoPropDerivatives(GpuInformation info, BenchmarkProperties prop) {

    // Create the final data structure.
    InfoPropDerivatives derivatives;

    // Create full cuda version.
    derivatives.cudaVersion = (((float) info.major) + (((float) info.minor) / 10.));

    // Create basic derivatives.
    // Data from "helper_cuda.h".
    if ((info.major == 3) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 3) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 3) && (info.minor == 5)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 3) && (info.minor == 7)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 5) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 5) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 5) && (info.minor == 3)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 6) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 6) && (info.minor == 1)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 6) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 7) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 7) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 7) && (info.minor == 5)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 8) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 8) && (info.minor == 6)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 8) && (info.minor == 7)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 8) && (info.minor == 9)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 9) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 128;
    } else {
        derivatives.numberOfCoresPerSm = 0;
    }
    derivatives.totalNumberOfCores = derivatives.numberOfCoresPerSm * info.multiProcessorCount;
    derivatives.hardwareWarpsPerSm = derivatives.numberOfCoresPerSm / info.warpSize;
    derivatives.maxNumberOfWarpsPerSm = info.maxThreadsPerMultiProcessor / info.warpSize;

    // Return the final data structure.
    return derivatives;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param prop The given data structure.
 */
void createInfoPropDerivativesFile(InfoPropDerivatives derivatives) {

    // Creation and opening of the csv file.
    char output[] = "raw/InfoProp_Derivatives.csv";
    FILE *csv = fopen(output, "w");

    // Writing the full cuda version into the csv file.
    fprintf(csv, "cudaVersion; \"%f\"\n", derivatives.cudaVersion);

    // Writing the basic derivatives into the csv file.
    fprintf(csv, "numberOfCoresPerSm;\"%d\"\n", derivatives.numberOfCoresPerSm);
    fprintf(csv, "totalNumberOfCores;\"%d\"\n", derivatives.totalNumberOfCores);
    fprintf(csv, "hardwareWarpsPerSm;\"%d\"\n", derivatives.hardwareWarpsPerSm);
    fprintf(csv, "maxNumberOfWarpsPerSm;\"%d\"\n", derivatives.maxNumberOfWarpsPerSm);

    // Close the csv file.
    fclose(csv);
    printf("[INFO] The info-prop-derivatives file was created.\n");
}


#endif //GCPVE_03_INFO_PROP_DERIVATIVES_CUH

//FINISHED

