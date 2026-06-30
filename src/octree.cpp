#include "octree.h"
#include <cmath>
#include <cstdlib>

OctreeNode::OctreeNode(float cx, float cy, float cz, float hs)
    : centerX(cx), centerY(cy), centerZ(cz), halfSize(hs),
      mass(0), comX(0), comY(0), comZ(0),
      body(nullptr), isLeaf(true) {
    for (int i = 0; i < 8; i++) children[i] = nullptr;
}

OctreeNode::~OctreeNode() {
    for (int i = 0; i < 8; i++) {
        if (children[i]) delete children[i];
    }
}

int getOctant(Body* b, float cx, float cy, float cz) {
    int oct = 0;
    if (b->x >= cx) oct |= 1;
    if (b->y >= cy) oct |= 2;
    if (b->z >= cz) oct |= 4;
    return oct;
}

void OctreeNode::insert(Body* b) {
    if (mass == 0) {
        // empty leaf - store body directly
        body = b;
        mass = b->mass;
        comX = b->x;
        comY = b->y;
        comZ = b->z;
        return;
    }

    if (isLeaf) {
        // already has a body, subdivide
        isLeaf = false;
        Body* oldBody = body;
        body = nullptr;

        float h = halfSize * 0.5f;
        for (int i = 0; i < 8; i++) {
            float newCx = centerX + (i & 1 ? h : -h);
            float newCy = centerY + (i & 2 ? h : -h);
            float newCz = centerZ + (i & 4 ? h : -h);
            children[i] = new OctreeNode(newCx, newCy, newCz, h);
        }

        int oct = getOctant(oldBody, centerX, centerY, centerZ);
        children[oct]->insert(oldBody);

        oct = getOctant(b, centerX, centerY, centerZ);
        children[oct]->insert(b);
    } else {
        // internal node, insert into correct child
        int oct = getOctant(b, centerX, centerY, centerZ);
        children[oct]->insert(b);
    }

    // update mass and center of mass
    mass += b->mass;
}

void OctreeNode::computeCenterOfMass() {
    if (isLeaf || mass == 0) return;

    comX = 0; comY = 0; comZ = 0;
    for (int i = 0; i < 8; i++) {
        if (children[i] && children[i]->mass > 0) {
            children[i]->computeCenterOfMass();
            comX += children[i]->mass * children[i]->comX;
            comY += children[i]->mass * children[i]->comY;
            comZ += children[i]->mass * children[i]->comZ;
        }
    }
    comX /= mass;
    comY /= mass;
    comZ /= mass;
}
