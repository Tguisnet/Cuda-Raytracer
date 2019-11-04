#pragma once

struct Object
{
    Object(int mesh, int mtl, Vector p, Vector r, Vector s)
    : mesh(mesh)
    , mtl(mtl)
    , pos(p)
    , rot(r)
    , scale(s)
    {}

    int mesh;
    int mtl;
    Vector pos;
    Vector rot;
    Vector scale;
};
