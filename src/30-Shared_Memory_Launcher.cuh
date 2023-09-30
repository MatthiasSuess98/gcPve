#ifndef GCPVE_30_SHARED_MEMORY_LANCHER_CUH
#define GCPVE_30_SHARED_MEMORY_LANCHER_CUH


/**
 *
 */
void launchSMBenchmark(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

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

            smallSMBenchmark<<<(4 * 30), (4 * 32)>>>(deviceLoad, deviceTime, i, j);
            cudaDeviceSynchronize();

            cudaMemcpy((void *) hostTime, (void *) deviceTime, (sizeof(float) * 32), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFreeHost(hostTime);
            cudaFreeHost(hostLoad);
            cudaFree(deviceTime);
            cudaFree(deviceLoad);
        }
    }
}

#endif //GCPVE_30_SHARED_MEMORY_LANCHER_CUH

//FINISHED

