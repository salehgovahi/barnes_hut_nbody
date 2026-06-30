#ifndef BARNES_HUT_HYBRID_CUH
#define BARNES_HUT_HYBRID_CUH

#include "body.h"

struct FlatNode {
    float mass;
    float comX, comY, comZ;
    float halfSize;
    int children[8];
    bool isLeaf;
    int bodyIndex;
};

__global__ void updatePositionsCUDA(Body* bodies, int n, float dt);

class BarnesHutHybrid {
public:
    Body* h_bodies;
    Body* d_bodies;
    int nBodies;
    float theta;
    float G;
    float dt;
    int threadsPerBlock;
    int numOMPThreads;

    BarnesHutHybrid(Body* bodies, int n, float theta, float G, float dt, int ompThreads);
    ~BarnesHutHybrid();

    void copyToDevice();
    void copyFromDevice();
    float run(int steps, bool verbose = true);
};

#endif