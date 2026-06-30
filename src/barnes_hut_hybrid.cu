#include "barnes_hut_hybrid.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <vector>

struct CPUNode {
    float cx, cy, cz, halfSize;
    float mass, comX, comY, comZ;
    int bodyIndex;
    int children[8];
    bool isLeaf;
};

std::vector<CPUNode> treePool;
int poolCounter = 0;

int allocNode(float cx, float cy, float cz, float hs) {
    if (poolCounter >= (int)treePool.size()) {
        treePool.resize(treePool.size() * 1.5);
    }
    int idx = poolCounter++;
    treePool[idx].cx = cx; treePool[idx].cy = cy; treePool[idx].cz = cz; treePool[idx].halfSize = hs;
    treePool[idx].mass = 0.0f; treePool[idx].comX = 0.0f; treePool[idx].comY = 0.0f; treePool[idx].comZ = 0.0f;
    treePool[idx].bodyIndex = -1;
    treePool[idx].isLeaf = true;
    for(int i=0; i<8; i++) treePool[idx].children[i] = -1;
    return idx;
}

int getOctantFast(float bx, float by, float bz, float cx, float cy, float cz) {
    int oct = 0;
    if (bx >= cx) oct |= 1;
    if (by >= cy) oct |= 2;
    if (bz >= cz) oct |= 4;
    return oct;
}

void insertFast(int nodeIdx, int bIdx, Body* bodies) {
    if (treePool[nodeIdx].mass == 0.0f) {
        treePool[nodeIdx].bodyIndex = bIdx;
        treePool[nodeIdx].mass = bodies[bIdx].mass;
        treePool[nodeIdx].comX = bodies[bIdx].x;
        treePool[nodeIdx].comY = bodies[bIdx].y;
        treePool[nodeIdx].comZ = bodies[bIdx].z;
        return;
    }

    if (treePool[nodeIdx].isLeaf) {
        treePool[nodeIdx].isLeaf = false;
        int oldBIdx = treePool[nodeIdx].bodyIndex;
        treePool[nodeIdx].bodyIndex = -1;

        float hs = treePool[nodeIdx].halfSize * 0.5f;
        float cx = treePool[nodeIdx].cx;
        float cy = treePool[nodeIdx].cy;
        float cz = treePool[nodeIdx].cz;

        for (int i = 0; i < 8; i++) {
            float ncx = cx + (i & 1 ? hs : -hs);
            float ncy = cy + (i & 2 ? hs : -hs);
            float ncz = cz + (i & 4 ? hs : -hs);
            treePool[nodeIdx].children[i] = allocNode(ncx, ncy, ncz, hs);
        }

        int oct1 = getOctantFast(bodies[oldBIdx].x, bodies[oldBIdx].y, bodies[oldBIdx].z, cx, cy, cz);
        insertFast(treePool[nodeIdx].children[oct1], oldBIdx, bodies);

        int oct2 = getOctantFast(bodies[bIdx].x, bodies[bIdx].y, bodies[bIdx].z, cx, cy, cz);
        insertFast(treePool[nodeIdx].children[oct2], bIdx, bodies);
    } else {
        int oct = getOctantFast(bodies[bIdx].x, bodies[bIdx].y, bodies[bIdx].z, treePool[nodeIdx].cx, treePool[nodeIdx].cy, treePool[nodeIdx].cz);
        insertFast(treePool[nodeIdx].children[oct], bIdx, bodies);
    }
    treePool[nodeIdx].mass += bodies[bIdx].mass;
}

void computeCOMFast_OMP(int nodeIdx) {
    if (treePool[nodeIdx].isLeaf || treePool[nodeIdx].mass == 0.0f) return;

    treePool[nodeIdx].comX = 0.0f;
    treePool[nodeIdx].comY = 0.0f;
    treePool[nodeIdx].comZ = 0.0f;

    for (int i = 0; i < 8; i++) {
        int childIdx = treePool[nodeIdx].children[i];
        if (childIdx != -1 && treePool[childIdx].mass > 0.0f) {
            if (treePool[nodeIdx].halfSize > 50.0f) {
                #pragma omp task shared(treePool)
                computeCOMFast_OMP(childIdx);
            } else {
                computeCOMFast_OMP(childIdx);
            }
        }
    }
    #pragma omp taskwait

    for (int i = 0; i < 8; i++) {
        int childIdx = treePool[nodeIdx].children[i];
        if (childIdx != -1 && treePool[childIdx].mass > 0.0f) {
            treePool[nodeIdx].comX += treePool[childIdx].mass * treePool[childIdx].comX;
            treePool[nodeIdx].comY += treePool[childIdx].mass * treePool[childIdx].comY;
            treePool[nodeIdx].comZ += treePool[childIdx].mass * treePool[childIdx].comZ;
        }
    }
    treePool[nodeIdx].comX /= treePool[nodeIdx].mass;
    treePool[nodeIdx].comY /= treePool[nodeIdx].mass;
    treePool[nodeIdx].comZ /= treePool[nodeIdx].mass;
}

__global__ void computeForcesCUDA_Tree(Body* bodies, FlatNode* tree, int n, float G, float theta, int rootIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float myX = bodies[idx].x;
    float myY = bodies[idx].y;
    float myZ = bodies[idx].z;

    int stack[64];
    int top = 0;
    stack[top++] = rootIdx;

    while (top > 0) {
        int nodeIdx = stack[--top];
        FlatNode node = tree[nodeIdx];

        if (node.mass == 0.0f) continue;

        float dx = node.comX - myX;
        float dy = node.comY - myY;
        float dz = node.comZ - myZ;
        float distSq = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(distSq + 0.01f);

        if (node.isLeaf) {
            if (node.bodyIndex != idx && node.bodyIndex != -1) {
                float f = G * node.mass / (distSq + 0.01f);
                fx += f * dx / dist;
                fy += f * dy / dist;
                fz += f * dz / dist;
            }
        } else {
            float s = node.halfSize * 2.0f;
            if (s / dist < theta) {
                float f = G * node.mass / (distSq + 0.01f);
                fx += f * dx / dist;
                fy += f * dy / dist;
                fz += f * dz / dist;
            } else {
                for (int i = 0; i < 8; i++) {
                    if (node.children[i] != -1) stack[top++] = node.children[i];
                }
            }
        }
    }

    bodies[idx].fx = fx;
    bodies[idx].fy = fy;
    bodies[idx].fz = fz;
}

__global__ void updatePositionsCUDA(Body* bodies, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bodies[idx].vx += bodies[idx].fx * dt;
    bodies[idx].vy += bodies[idx].fy * dt;
    bodies[idx].vz += bodies[idx].fz * dt;

    bodies[idx].x += bodies[idx].vx * dt;
    bodies[idx].y += bodies[idx].vy * dt;
    bodies[idx].z += bodies[idx].vz * dt;
}

BarnesHutHybrid::BarnesHutHybrid(Body* bodies, int n, float theta, float G, float dt, int ompThreads)
    : h_bodies(bodies), nBodies(n), theta(theta), G(G), dt(dt),
      threadsPerBlock(256), numOMPThreads(ompThreads) {
    cudaMalloc(&d_bodies, n * sizeof(Body));
}

BarnesHutHybrid::~BarnesHutHybrid() {
    cudaFree(d_bodies);
}

void BarnesHutHybrid::copyToDevice() {
    cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice);
}

void BarnesHutHybrid::copyFromDevice() {
    cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost);
}

float BarnesHutHybrid::run(int steps, bool verbose) {
    int blocksGPU = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

    omp_set_num_threads(numOMPThreads);

    if (verbose) {
        std::cout << "  [Hybrid] CPU(OpenMP Tasks) for Tree & GPU(CUDA) for Forces" << std::endl;
    }

    int maxTreeNodes = nBodies * 8;

    FlatNode* d_tree;
    cudaMalloc(&d_tree, maxTreeNodes * sizeof(FlatNode));

    treePool.resize(maxTreeNodes);
    std::vector<FlatNode> flatTree(maxTreeNodes);

    copyToDevice();

    double start = omp_get_wtime();
    for (int s = 0; s < steps; s++) {

        poolCounter = 0;
        int rootIdx = allocNode(0, 0, 0, 1000.0f);
        for (int i = 0; i < nBodies; i++) {
            insertFast(rootIdx, i, h_bodies);
        }

        #pragma omp parallel
        {
            #pragma omp single
            {
                computeCOMFast_OMP(rootIdx);
            }
        }

        if ((int)flatTree.size() < poolCounter) {
            flatTree.resize(treePool.size());
        }

        for(int i = 0; i < poolCounter; i++) {
            flatTree[i].mass = treePool[i].mass;
            flatTree[i].comX = treePool[i].comX;
            flatTree[i].comY = treePool[i].comY;
            flatTree[i].comZ = treePool[i].comZ;
            flatTree[i].halfSize = treePool[i].halfSize;
            flatTree[i].isLeaf = treePool[i].isLeaf;
            flatTree[i].bodyIndex = treePool[i].bodyIndex;
            for(int c=0; c<8; c++) flatTree[i].children[c] = treePool[i].children[c];
        }

        cudaMemcpy(d_tree, flatTree.data(), poolCounter * sizeof(FlatNode), cudaMemcpyHostToDevice);

        computeForcesCUDA_Tree<<<blocksGPU, threadsPerBlock>>>(d_bodies, d_tree, nBodies, G, theta, rootIdx);
        cudaDeviceSynchronize();

        updatePositionsCUDA<<<blocksGPU, threadsPerBlock>>>(d_bodies, nBodies, dt);
        cudaDeviceSynchronize();

        copyFromDevice();

        if (verbose && s % 10 == 0) {
            std::cout << "  [Hybrid] Step " << s << "/" << steps << " done\r" << std::flush;
        }
    }
    double end = omp_get_wtime();

    cudaFree(d_tree);

    float elapsed = (float)(end - start);
    if (verbose) {
        std::cout << "\n  [Hybrid] Completed " << steps << " steps in " << std::fixed << std::setprecision(4) << elapsed << " seconds" << std::endl;
    }
    return elapsed;
}