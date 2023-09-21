#ifndef GCPVE_04_DATA_COLLECTION_CUH
#define GCPVE_04_DATA_COLLECTION_CUH

/**
 * Data structure for small data collections.
 */
typedef struct SmallDataCollection {
    // small = 16 bit
    int mulp[65536];
    int warp[65536];
    int lane[65536];
    float time[65536];
} SmallDataCollection;


/**
 * Data structure for medium data collections.
 */
typedef struct MediumDataCollection {
    // medium = 24 bit
    int mulp[16777216];
    int warp[16777216];
    int lane[16777216];
    float time[16777216];
} MediumDataCollection;


/**
 * Data structure for large data collections.
 */
typedef struct LargeDataCollection {
    // large = 32 bit
    int mulp[4294967296];
    int warp[4294967296];
    int lane[4294967296];
    float time[4294967296];
} LargeDataCollection;

#endif //GCPVE_04_DATA_COLLECTION_CUH

//FINISHED

