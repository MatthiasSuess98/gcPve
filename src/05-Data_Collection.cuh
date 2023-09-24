#ifndef GCPVE_04_DATA_COLLECTION_CUH
#define GCPVE_04_DATA_COLLECTION_CUH

#include <vector>

/**
 * Data structure for small data collections.
 */
typedef struct SmallDataCollection {
    // small = 16 bit
    int mulp[65536];
    int warp[65536];
    int lane[65536];
    long long int time[65536];
} SmallDataCollection;


/**
 * Data structure for medium data collections.
 */
typedef struct MediumDataCollection {
    // medium = 24 bit
    int mulp[16777216];
    int warp[16777216];
    int lane[16777216];
    long long int time[16777216];
} MediumDataCollection;


/**
 * Data structure for large data collections.
 */
typedef struct LargeDataCollection {
    // large = 32 bit
    int mulp[4294967296];
    int warp[4294967296];
    int lane[4294967296];
    long long int time[4294967296];
} LargeDataCollection;

#endif //GCPVE_04_DATA_COLLECTION_CUH

//FINISHED

