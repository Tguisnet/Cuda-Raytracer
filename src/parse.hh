#pragma once

#include "material.hh"
#include "triangle.hh"
#include "scene.hh"

#include <unordered_map>

void obj_to_vertices(const std::string &s, const std::vector <std::string> &mat_names,
                     std::vector<Triangle>& v_tri, Scene &scene);
void parse_materials(const std::string &s, Scene &scene);

int write_ppm(const std::string &out_path, const Pixel *pix,
              int width, int height);

Scene parse_scene(const std::string& filename);
