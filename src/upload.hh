#pragma once

#include "kdtree.hh"
#include "kdtree_gpu.hh"
#include "triangle.hh"

#define cudaCheckError(ans) gpuAssert((ans), __FILE__, __LINE__)
void gpuAssert(cudaError_t code, const char *file, int line);

KdNodeGpu* upload_kd_tree(const KdTree& kd_tree, std::vector<Triangle>& triangles);
