#pragma once

#include "triangle.hh"

struct KdNodeGpu
{
    KdNodeGpu *left;
    KdNodeGpu *right;

    float box[6];

    Triangle *beg;
    Triangle *end;
};

struct Material;
struct Light;

__device__ Pixel direct_light(const KdNodeGpu *root, Ray &r, const Material *materials,
                              const Vector *a_light, const Light *d_lights);
