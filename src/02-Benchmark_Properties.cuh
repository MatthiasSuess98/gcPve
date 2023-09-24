#ifndef GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH
#define GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

/**
 * Data structure for all properties of the benchmark.
 */
typedef struct BenchmarkProperties {

    // Variables.
    int numberOfTrialsPerform;
    int numberOfTrialsLaunch;
    int memoryOverlap;
    long double maxDelta;
    int maxDontFit;
    int small;
    int medium;
    long large;
    int numberOfTrialsDivisor;
} BenchmarkProperties;


/**
 * Creates a data structure with all properties of the benchmark.
 * @return Data structure with all properties of the benchmark.
 */
BenchmarkProperties getBenchmarkProperties() {

    // Create the final data structure.
    BenchmarkProperties prop;

    // Initialize the properties and writes them into the final data structure.
    prop.numberOfTrialsPerform = 10;
    prop.numberOfTrialsLaunch = 10;
    prop.memoryOverlap = 2;
    prop.maxDelta = 8.0;
    prop.maxDontFit = 1;
    // Size of the data collections.
    // Warning: If these three variables get updated, update the variables in 04-Core_Characteristics and in the kernels also!
    prop.small = 65536;
    prop.medium = 16777216;
    prop.large = 4294967296;
    // Warning: If this variable gets updated, update the variables in the kernels also!
    prop.numberOfTrialsDivisor = 1024;

    // Return the final data structure.
    return prop;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param prop The given data structure.
 */
void createPropFile(BenchmarkProperties prop) {

    // Creation and opening of the csv file.
    char output[] = "raw/Bench_Prop.csv";
    FILE *csv = fopen(output, "w");

    // Writing all the information into the csv file.
    fprintf(csv, "numberOfTrialsPerform; \"%d\"\n", prop.numberOfTrialsPerform);
    fprintf(csv, "numberOfTrialsLaunch; \"%d\"\n", prop.numberOfTrialsLaunch);
    fprintf(csv, "memoryOverlap; \"%d\"\n", prop.memoryOverlap);
    fprintf(csv, "maxDelta; \"%Lf\"\n", prop.maxDelta);
    fprintf(csv, "maxDontFit; \"%d\"\n", prop.maxDontFit);
    fprintf(csv, "small; \"%d\"\n", prop.small);
    fprintf(csv, "medium; \"%d\"\n", prop.medium);
    fprintf(csv, "large; \"%ld\"\n", prop.large);
    fprintf(csv, "numberOfTrialsDivisor; \"%d\"\n", prop.numberOfTrialsDivisor);

    // Close the csv file.
    fclose(csv);
    printf("[INFO] The benchmark properties file was created.\n");
}

#endif //GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

//FINISHED

