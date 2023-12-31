#ifndef GCPVE_04_CORE_CHARACTERISTICS_CUH
#define GCPVE_04_CORE_CHARACTERISTICS_CUH


/**
 * Object class of the core characteristics.
 */
class CoreCharacteristics {
private:
    int smId;
    int hardwareWarpId;
    int warpCoreId;
    float typicalL1Time;
    float typicalSmTime;
    float typicalL2Time;
    float typicalGmTime;

public:
    CoreCharacteristics(int newSmId, int newHardwareWarpId, int newWarpCoreId) {
        smId = newSmId;
        hardwareWarpId = newHardwareWarpId;
        warpCoreId = newWarpCoreId;
        typicalL1Time = 0.0;
        typicalSmTime = 0.0;
        typicalL2Time = 0.0;
        typicalGmTime = 0.0;
    }

    void setTypicalL1Time(float newTypicalL1Time) {
        typicalL1Time = newTypicalL1Time;
    }

    void setTypicalSmTime(float newTypicalSmTime) {
        typicalSmTime = newTypicalSmTime;
    }

    void setTypicalL2Time(float newTypicalL2Time) {
        typicalL2Time = newTypicalL2Time;
    }

    void setTypicalGmTime(float newTypicalGmTime) {
        typicalGmTime = newTypicalGmTime;
    }

    int getSmId() {
        return smId;
    }

    int getHardwareWarpId() {
        return hardwareWarpId;
    }

    int getWarpCoreId() {
        return warpCoreId;
    }

    float getTypicalL1Time() {
        return typicalL1Time;
    }

    float getTypicalSmTime() {
        return typicalSmTime;
    }

    float getTypicalL2Time() {
        return typicalL2Time;
    }

    float getTypicalGmTime() {
        return typicalGmTime;
    }
};


#endif //GCPVE_04_CORE_CHARACTERISTICS_CUH

//FINISHED

