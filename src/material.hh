#pragma once

#include "vector.hh"

struct Material
{
    Material() = default;
    Material(float ns, Vector &ka, Vector &kd, Vector &ks,
             Vector &ke, float ni, float d, int illum, Vector &tf)
        : ns(ns)
        , ka(ka)
        , kd(kd)
        , ks(ks)
        , ke(ke)
        , ni(ni)
        , d(d)
        , illum(illum)
        , tf(tf)
    { }

    void dump();

    float ns;
    Vector ka;
    Vector kd;
    Vector ks;
    Vector ke;
    float ni;
    float d;
    int illum;
    Vector tf;
};
