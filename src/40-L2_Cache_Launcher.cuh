#ifndef GCPVE_40_L2_CACHE_LAUNCHER_CUH
#define GCPVE_40_L2_CACHE_LAUNCHER_CUH


/**
 *
 */
void launchL2Benchmark(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    char output[] = "raw/Benchmark_L2.csv";
    FILE *csv = fopen(output, "w");
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 20; j++) {

            float *hostTime = nullptr;
            cudaMallocHost((void **) &hostTime, (sizeof(float) * 32));
            unsigned int *hostLoad = nullptr;
            cudaMallocHost((void **) &hostLoad, (sizeof(unsigned int) * 1024));
            float *deviceTime = nullptr;
            cudaMalloc((void **) &deviceTime, (sizeof(float) * 32));
            unsigned int *deviceLoad = nullptr;
            cudaMalloc((void **) &deviceLoad, (sizeof(unsigned int) * 1024));

            for (int k = 0; k < 1024; k++) {
                hostLoad[k] = (k * 512) % 1024;
            }

            cudaMemcpy((void *) deviceLoad, (void *) hostLoad, (sizeof(unsigned int) * 1024), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            smallL2Benchmark<<<(4 * 30), (4 * 32)>>>(deviceLoad, deviceTime, i, j);
            cudaDeviceSynchronize();

            cudaMemcpy((void *) hostTime, (void *) deviceTime, (sizeof(float) * 32), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            for (int k = 0; k < 32; k++) {
                fprintf(csv, "%f", hostTime[k]);
                if (k < (32 - 1)) {
                    fprintf(csv, ";");
                }
            }

            cudaFreeHost(hostTime);
            cudaFreeHost(hostLoad);
            cudaFree(deviceTime);
            cudaFree(deviceLoad);

            fprintf(csv, "\n");
        }
        fprintf(csv, "\n");
    }
    fclose(csv);
}


#endif //GCPVE_40_L2_CACHE_LAUNCHER_CUH

//FINISHED

