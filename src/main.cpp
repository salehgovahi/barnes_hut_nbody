#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <vector>
#include "body.h"

// Forward declarations for the available implementations
float runSerial(Body* bodies, int n, float theta, float G, float dt, int steps);

// CUDA/Hybrid availability check
bool cudaAvailable();
float runHybrid(Body* bodies, int n, float theta, float G, float dt, int steps, int ompThreads);

void initBodies(Body* bodies, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        bodies[i].x = (float)(rand() % 2000 - 1000);
        bodies[i].y = (float)(rand() % 2000 - 1000);
        bodies[i].z = (float)(rand() % 2000 - 1000);
        bodies[i].vx = 0;
        bodies[i].vy = 0;
        bodies[i].vz = 0;
        bodies[i].mass = (float)(rand() % 100 + 10);
        bodies[i].fx = 0;
        bodies[i].fy = 0;
        bodies[i].fz = 0;
    }
}

void copyBodies(Body* src, Body* dst, int n) {
    memcpy(dst, src, n * sizeof(Body));
}

void printBanner() {
    std::cout << "+========================================================================+" << std::endl;
    std::cout << "|        Barnes-Hut N-Body Simulation - Performance Comparison           |" << std::endl;
    std::cout << "|                          Serial | Hybrid                               |" << std::endl;
    std::cout << "+========================================================================+" << std::endl;
    std::cout << std::endl;
}

void printSystemInfo() {
    std::cout << "+------------------------------------------------------------------------+" << std::endl;
    std::cout << "| System Information:                                                      |" << std::endl;
    std::cout << "+------------------------------------------------------------------------+" << std::endl;

    #ifdef _OPENMP
    std::cout << "|  OpenMP:        Enabled (Version " << _OPENMP << ")" << std::endl;
    std::cout << "|  CPU Threads:   " << omp_get_max_threads() << " available" << std::endl;
    #else
    std::cout << "|  OpenMP:        Not Available" << std::endl;
    #endif

    bool hasCuda = cudaAvailable();
    if (hasCuda) {
        std::cout << "|  CUDA:          Available" << std::endl;
    } else {
        std::cout << "|  CUDA:          Not Available (Hybrid mode requires CUDA)" << std::endl;
    }
    std::cout << "+------------------------------------------------------------------------+" << std::endl;
    std::cout << std::endl;
}

void printResults(float serialTime, float hybridTime, int nBodies, int steps) {
    std::cout << std::endl;
    std::cout << "+========================================================================+" << std::endl;
    std::cout << "|                         PERFORMANCE RESULTS                              |" << std::endl;
    std::cout << "+========================================================================+" << std::endl;
    std::cout << "|  Simulation: " << std::setw(6) << nBodies << " bodies, " << std::setw(4) << steps << " steps                                 |" << std::endl;
    std::cout << "+========================================================================+" << std::endl;

    std::cout << "|  BASELINE (Serial):                                                     |" << std::endl;
    std::cout << "|    Time: " << std::setw(10) << std::fixed << std::setprecision(4) << serialTime << " seconds                                      |" << std::endl;
    std::cout << "|    Speedup: 1.00x (baseline)                                            |" << std::endl;
    std::cout << "+========================================================================+" << std::endl;

    if (hybridTime > 0) {
        float speedup = serialTime / hybridTime;
        std::cout << "|  HYBRID (CPU Tree + GPU Compute) Results:                               |" << std::endl;
        std::cout << "|    Time: " << std::setw(10) << std::fixed << std::setprecision(4) << hybridTime << " seconds                                      |" << std::endl;
        std::cout << "|    Speedup: " << std::setw(6) << std::setprecision(2) << speedup << "x                                              |" << std::endl;
        std::cout << "+========================================================================+" << std::endl;
    } else {
        std::cout << "|  HYBRID: skipped (no CUDA device found)                                |" << std::endl;
        std::cout << "+========================================================================+" << std::endl;
    }
}

int main(int argc, char** argv) {
    printBanner();
    printSystemInfo();

    int nBodies = 5000;
    int steps = 50;
    float theta = 0.5f;
    float G = 6.674f;
    float dt = 0.01f;

    if (argc > 1) nBodies = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);

    std::cout << "Configuration: " << nBodies << " bodies, " << steps << " steps" << std::endl;
    std::cout << "Theta: " << theta << ", G: " << G << ", dt: " << dt << std::endl;
    std::cout << std::endl;

    Body* originalBodies = new Body[nBodies];
    Body* workingBodies = new Body[nBodies];
    initBodies(originalBodies, nBodies);

    int maxThreads = omp_get_max_threads();
    float hybridTime = 0;

    // ========== SERIAL ==========
    std::cout << "===========================================================================" << std::endl;
    std::cout << "Running SERIAL implementation..." << std::endl;
    copyBodies(originalBodies, workingBodies, nBodies);
    float serialTime = runSerial(workingBodies, nBodies, theta, G, dt, steps);

    // ========== HYBRID ==========
    bool hasCuda = cudaAvailable();
    if (hasCuda) {
        std::cout << std::endl;
        std::cout << "===========================================================================" << std::endl;
        std::cout << "Running HYBRID (CUDA + OpenMP) implementation..." << std::endl;

        copyBodies(originalBodies, workingBodies, nBodies);
        hybridTime = runHybrid(workingBodies, nBodies, theta, G, dt, steps, maxThreads);
    } else {
        std::cout << std::endl;
        std::cout << "CUDA not available, skipping Hybrid test." << std::endl;
    }

    // ========== RESULTS ==========
    printResults(serialTime, hybridTime, nBodies, steps);

    delete[] originalBodies;
    delete[] workingBodies;

    std::cout << std::endl;
    std::cout << "All tests completed!" << std::endl;

    return 0;
}