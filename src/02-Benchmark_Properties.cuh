#ifndef GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH
#define GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

/**
 * Data structure for all properties of the benchmark.
 */
typedef struct BenchmarkProperties {

    // Variables.
    int numberOfTrialsPerform;
    int numberOfTrialsLaunch;
    int numberOfTrialsBenchmark;
    int memoryOverlap;
    int maxDelta;
    int maxDontFit;
} BenchmarkProperties;


/**
 * Creates a data structure with all properties of the benchmark.
 * @return Data structure with all properties of the benchmark.
 */
BenchmarkProperties getBenchmarkProperties() {

    // Create the final data structure.
    BenchmarkProperties prop;

    // Initialize the properties and writes them into the final data structure.
    prop.numberOfTrialsPerform = 1;
    prop.numberOfTrialsLaunch = 10;
    prop.numberOfTrialsBenchmark = 1024;
    prop.memoryOverlap = 2;
    prop.maxDelta = 0;
    prop.maxDontFit = 3;

    // Return the final data structure.
    return prop;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param prop The given data structure.
 */
void createPropFile(BenchmarkProperties prop) {

    // Creation and opening of the csv file.
    char output[] = "BenchProp.csv";
    FILE *csv = fopen(output, "w");

    // Writing all the information into the csv file.
    fprintf(csv, "numberOfTrialsPerform; \"%d\"\n", prop.numberOfTrialsPerform);
    fprintf(csv, "numberOfTrialsLaunch; \"%d\"\n", prop.numberOfTrialsLaunch);
    fprintf(csv, "numberOfTrialsBenchmark; \"%d\"\n", prop.numberOfTrialsBenchmark);
    fprintf(csv, "memoryOverlap; \"%d\"\n", prop.memoryOverlap);
    fprintf(csv, "maxDelta; \"%d\"\n", prop.maxDelta);
    fprintf(csv, "maxDontFit; \"%d\"\n", prop.maxDontFit);

    // Close the csv file.
    fclose(csv);
    printf("[INFO] Benchmark properties file created.\n");
}

#endif //GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

//FINISHED

