#pragma once

#include "light.hh"
#include "material.hh"
#include "matrix.hh"
#include "object.hh"

#include <vector>
#include <unordered_map>

struct Scene
{
    int height;
    int width;
    Vector cam_pos;
    Vector cam_u;
    Vector cam_v;
    float fov;
    std::vector<std::string> objs;
    std::vector<std::string> mtls;
    Vector a_light;
    Matrix transform;
    std::vector<Light> lights;
    std::vector<Object> objects;
    std::vector<Vector> emissive; // coord of emmisive lights
    std::vector<std::string> emissive_name;

    std::vector<std::string> mat_names;
    Material *materials;
    std::unordered_map<std::string, Material> map;
};
