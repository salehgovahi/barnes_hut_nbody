#include "body.h"
#include "barnes_hut_serial.h"
#include "barnes_hut_hybrid.cuh"
#include <cuda_runtime.h>
#include <iostream>

bool cudaAvailable() {
    int cudaDevices = 0;
    cudaError_t err = cudaGetDeviceCount(&cudaDevices);
    return (err == cudaSuccess && cudaDevices > 0);
}

float runSerial(Body* bodies, int n, float theta, float G, float dt, int steps) {
    BarnesHutSerial sim(bodies, n, theta, G, dt);
    return sim.run(steps);
}

float runHybrid(Body* bodies, int n, float theta, float G, float dt, int steps, int ompThreads) {
    if (!cudaAvailable()) return 0.0f;
    BarnesHutHybrid sim(bodies, n, theta, G, dt, ompThreads);
    return sim.run(steps);
}