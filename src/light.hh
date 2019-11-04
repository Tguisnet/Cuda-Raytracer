#pragma once

#include <iostream>
#include "kdtree.hh"
#include "triangle.hh"
#include "vector.hh"

#define MAX_LIGHTS (static_cast<size_t>(8))

struct Light // directional
{
    Light(const Vector &color,
          const Vector &dir)
    : color(color)
    , dir(dir)
    {}

    virtual ~Light() = default;

    Vector color;
    Vector dir;
};

std::ostream& operator <<(std::ostream& os, const Light &l);
