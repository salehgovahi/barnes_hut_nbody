#ifndef BARNES_HUT_SERIAL_H
#define BARNES_HUT_SERIAL_H

#include "body.h"
#include "octree.h"

class BarnesHutSerial {
public:
    Body* bodies;
    int nBodies;
    float theta;
    float G;
    float dt;
    float halfSize;

    BarnesHutSerial(Body* bodies, int n, float theta, float G, float dt);
    ~BarnesHutSerial();

    void buildOctree();
    void computeForce(Body* body, OctreeNode* node, float& fx, float& fy, float& fz);
    void computeForces();
    void updatePositions();
    void step();
    float run(int steps, bool verbose = true);

private:
    OctreeNode* root;
};

#endif
