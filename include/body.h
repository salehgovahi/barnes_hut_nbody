#ifndef BODY_H
#define BODY_H

struct Body {
    float x, y, z;      // position
    float vx, vy, vz;   // velocity
    float mass;         // mass
    float fx, fy, fz;   // force
};

#endif
