#pragma once

#include <cfloat>

#include "vector.hh"

#define EPSILON FLT_EPSILON
#define BIAS    0.001

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

struct Ray;

struct Triangle
{
    Triangle(const Vector &v1, const Vector &v2, const Vector &v3,
             const Vector &n1, const Vector &n2, const Vector &n3,
             unsigned char id)
    {
        vertices[0] = v1;
        vertices[1] = v2;
        vertices[2] = v3;

        normal[0] = n1;
        normal[1] = n2;
        normal[2] = n3;

        this->id = id;
    }

    Triangle() = default;

    Vector get_mean(void) const // return barycentre
    {
        float x = 0.f;
        float y = 0.f;
        float z = 0.f;

        for (unsigned i = 0; i < 3; ++i)
        {
            x += vertices[i][0];
            y += vertices[i][1];
            z += vertices[i][2];
        }

        return Vector(x / 3.f, y / 3.f, z / 3.f);
    }

    CUDA_HOSTDEV bool intersect(Ray &ray, float &dist) const;

    Vector vertices[3];
    Vector normal[3];
    unsigned char id;
};

struct Ray
{
    CUDA_HOSTDEV Ray(const Vector &o, const Vector &dir) : o(o), dir(dir)
    {
        inv = Vector(1.f / dir[0], 1.f / dir[1], 1.f / dir[2]);
        sign[0] = inv[0] < 0;
        sign[1] = inv[1] < 0;
        sign[2] = inv[2] < 0;
    };
    const Vector &o;
    const Vector &dir;
    Vector inv;
    Triangle *tri;

    float u;
    float v;

    unsigned char sign[3];
};
