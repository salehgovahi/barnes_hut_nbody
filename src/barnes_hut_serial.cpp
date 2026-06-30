#include "barnes_hut_serial.h"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <iomanip>

BarnesHutSerial::BarnesHutSerial(Body* bodies, int n, float theta, float G, float dt)
    : bodies(bodies), nBodies(n), theta(theta), G(G), dt(dt), halfSize(1000.0f), root(nullptr) {
}

BarnesHutSerial::~BarnesHutSerial() {
    if (root) delete root;
}

void BarnesHutSerial::buildOctree() {
    if (root) delete root;
    root = new OctreeNode(0, 0, 0, halfSize);
    for (int i = 0; i < nBodies; i++) {
        root->insert(&bodies[i]);
    }
    root->computeCenterOfMass();
}

void BarnesHutSerial::computeForce(Body* body, OctreeNode* node, float& fx, float& fy, float& fz) {
    if (!node || node->mass == 0) return;
    if (node->isLeaf && node->body == body) return;

    float dx = node->comX - body->x;
    float dy = node->comY - body->y;
    float dz = node->comZ - body->z;
    float distSq = dx*dx + dy*dy + dz*dz;
    float dist = sqrtf(distSq + 0.01f);

    if (node->isLeaf) {
        float f = G * node->mass / (distSq + 0.01f);
        fx += f * dx / dist;
        fy += f * dy / dist;
        fz += f * dz / dist;
    } else {
        float s = node->halfSize * 2.0f;
        if (s / dist < theta) {
            float f = G * node->mass / (distSq + 0.01f);
            fx += f * dx / dist;
            fy += f * dy / dist;
            fz += f * dz / dist;
        } else {
            for (int i = 0; i < 8; i++) {
                if (node->children[i]) {
                    computeForce(body, node->children[i], fx, fy, fz);
                }
            }
        }
    }
}

void BarnesHutSerial::computeForces() {
    for (int i = 0; i < nBodies; i++) {
        float fx = 0, fy = 0, fz = 0;
        computeForce(&bodies[i], root, fx, fy, fz);
        bodies[i].fx = fx;
        bodies[i].fy = fy;
        bodies[i].fz = fz;
    }
}

void BarnesHutSerial::updatePositions() {
    for (int i = 0; i < nBodies; i++) {
        bodies[i].vx += bodies[i].fx * dt;
        bodies[i].vy += bodies[i].fy * dt;
        bodies[i].vz += bodies[i].fz * dt;

        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

void BarnesHutSerial::step() {
    buildOctree();
    computeForces();
    updatePositions();
}

float BarnesHutSerial::run(int steps, bool verbose) {
    double start = omp_get_wtime();
    for (int s = 0; s < steps; s++) {
        step();
        if (verbose && s % 10 == 0) {
            std::cout << "  [Serial] Step " << s << "/" << steps << " done\r" << std::flush;
        }
    }
    double end = omp_get_wtime();
    float elapsed = (float)(end - start);
    if (verbose) {
        std::cout << "\n  [Serial] Completed " << steps << " steps in " << std::fixed << std::setprecision(4) << elapsed << " seconds" << std::endl;
    }
    return elapsed;
}
