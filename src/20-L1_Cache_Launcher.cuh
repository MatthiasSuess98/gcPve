#ifndef GCPVE_COLLECTDATA_CUH
#define GCPVE_COLLECTDATA_CUH

#define maximumNumberOfTrials 16

/**
 * Function that performs a L1 benchmark for small data collections.
 * It uses a shotgun technique
 * @param gpuInfo All available information of the current GPU.
 * @param benProp All properties of the benchmarks.
 * @return A complete fully sorted SmallDataCollection for the L1 cache.
 */
SmallDataCollection performSmallL1Benchmark (GpuInformation gpuInfo, BenchmarkProperties benProp) {

    // gpuInfo and benProp derivatives.
    numberOfBlocks = benProp.small / gpuInfo.warpSize;
    numberOfBlocksPerMulp = numberOfBlocks / gpuInfo.multiProcessorCount;

    // Collection A: For host use only.
    SmallDataCollection finalCollection;

    // Collection B: For both host use and device use.
    SmallDataCollection benchCollection;

    // Allocation of collection B to the global device memory.
    SmallDataCollection *ptr;
    ptr = &collection;
    cudaMallocManaged(&ptr, sizeof(collection));

    // Loop variables.
    bool moveOn = true;
    int numberOfTrials = 0;
    int numberOfSets = 0;

    // Multiprocessor loop.
    for (int i = 0; i < gpuInfo.multiProcessorCount; i++) {

        // Trail loop.
        while (moveOn) {

            // Check whether the maximum of trails is reached.
            if (numberOfTrials < maximumNumberOfTrials) {

                // Launch a trail.
                numberOfTrials++;
                launchL1Benchmarks(ptr, numberOfBlocks);
                cudaDeviceSynchronize();

                // Analyze the trail by checking all launched blocks.
                for (int j = 0; j < numberOfBlocks; j = j++) {

                    // Check whether the block is relevant.
                    if (moveOn && ((*ptr).time[j] != 0)) {

                        // Copy the data of this block into collection A.
                        finalCollection.mulp[(i * numberOfSetsPerMulp) + numberOfSets] = (*ptr).mulp[j];
                        finalCollection.warp[(i * numberOfSetsPerMulp) + numberOfSets] = (*ptr).warp[j];
                        finalCollection.lane[(i * numberOfSetsPerMulp) + numberOfSets] = (*ptr).lane[j];
                        finalCollection.time[(i * numberOfSetsPerMulp) + numberOfSets] = (*ptr).time[j];
                        numberOfSets++;

                        // Check whether all available sets are used.
                        if (numberOfSets >= numberOfSetsPerMulp) {
                            moveOn = false;
                        }
                    }
                }

            // If the maximum of trails is reached and some sets could not be filled: Print error.
            } else {
                moveOn = false;
                printf("[ERROR] Failed to get full l1 data for streaming multiprocessor %d in small benchmark.", i);
            }
        }
    }

    // Free the allocated global device memory of collection B.
    cudaFree(ptr);

    // Fill the remaining sets of collection A with void data.
    for (int i = numberOfSetsPerMulp * gpuInfo.multiProcessorCount; i < numberOfBlocks; i++) {
        finalCollection.mulp[i] = 0;
        finalCollection.warp[i] = 0;
        finalCollection.lane[i] = 0;
        finalCollection.time[i] = 0;
    }

    //Return the data collection A with the final benchmark data.
    return finalCollection;
}




#endif //GCPVE_COLLECTDATA_CUH

