#ifndef GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH
#define GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

/**
 * Data structure for all properties of the benchmark.
 */
typedef struct BenchmarkProperties {

    // Variables.
    unsigned int small;
    unsigned int medium;
    unsigned int large;
    unsigned int maximumNumberOfTrials;
} BenchmarkProperties;


/**
 * Creates a data structure with all properties of the benchmark.
 * @return Data structure with all properties of the benchmark.
 */
BenchmarkProperties getBenchmarkProperties() {

    // Create the final data structure.
    BenchmarkProperties prop;

    // Initialize the properties and writes them into the final data structure.
    prop.maximumNumberOfTrials = 16;
    // Size of the data collections.
    // Warning: If these three variables get updated, update the variables in 04-Core_Characteristics also!
    prop.small = 65536;
    prop.medium = 16777216;
    prop.large = 4294967296;

    // Return the final data structure.
    return info;
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
    fprintf(csv, "small; \"%d\"\n", prop.small);
    fprintf(csv, "medium; \"%d\"\n", prop.medium);
    fprintf(csv, "large; \"%d\"\n", prop.large);
    fprintf(csv, "maximumNumberOfTrials; \"%d\"\n", prop.maximumNumberOfTrials);

    // Close the csv file.
    fclose(csv);
}

#endif //GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

//FINISHED

