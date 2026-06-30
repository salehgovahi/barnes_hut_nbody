#ifndef OCTREE_H
#define OCTREE_H

#include "body.h"

struct OctreeNode {
    float centerX, centerY, centerZ;  // center of this node
    float halfSize;                   // half the size of this cube
    float mass;                       // total mass of this node
    float comX, comY, comZ;           // center of mass
    Body* body;                       // pointer to body if leaf (single body)
    OctreeNode* children[8];          // 8 octants
    bool isLeaf;

    OctreeNode(float cx, float cy, float cz, float hs);
    ~OctreeNode();
    void insert(Body* b);
    void computeCenterOfMass();
};

#endif
