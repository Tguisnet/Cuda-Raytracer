#include <stdio.h>

#include "upload.hh"

void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess)
        return;
    fprintf(stderr, "CUDA: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
}

static size_t upload_kd_node(KdNodeGpu* nodes, KdNodeGpu* nodes_gpu, 
                             std::vector<Triangle>& triangles, Triangle* triangles_gpu,
                             const KdTree::childPtr& kd_node, std::size_t& idx)
{
    if (!kd_node)
        return 0;

    size_t cur_idx = idx;

    KdNodeGpu &node = nodes[idx++];

    size_t left_idx = upload_kd_node(nodes, nodes_gpu, triangles,
                                     triangles_gpu, kd_node->left, idx);

    if (left_idx)
        node.left = nodes_gpu + left_idx;
    else
        node.left = nullptr;

    size_t right_idx = upload_kd_node(nodes, nodes_gpu, triangles,
                                     triangles_gpu, kd_node->right, idx);

    if (right_idx)
        node.right = nodes_gpu + right_idx;
    else
        node.right = nullptr;

    memcpy(node.box, kd_node->box, sizeof(node.box));

    size_t len = kd_node->end - kd_node->beg;
    size_t offset = &(*kd_node->beg) - triangles.data();

    node.beg = triangles_gpu + offset;
    node.end = node.beg + len;

    return cur_idx;
}

KdNodeGpu* upload_kd_tree(const KdTree& kd_tree, std::vector<Triangle>& triangles)
{
    std::vector<KdNodeGpu> nodes(kd_tree.nodes_count_);
    KdNodeGpu* nodes_gpu;
    cudaCheckError(cudaMalloc(&nodes_gpu, sizeof(*nodes_gpu) * nodes.size()));
    Triangle* triangles_gpu;
    cudaCheckError(cudaMalloc(&triangles_gpu, sizeof(*triangles_gpu) * triangles.size()));
    size_t idx = 0;

    upload_kd_node(nodes.data(), nodes_gpu, triangles,
                   triangles_gpu,
                   kd_tree.root_, 
                   idx);

    cudaCheckError(cudaMemcpy(nodes_gpu, nodes.data(), sizeof(*nodes_gpu) * nodes.size(), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(triangles_gpu, triangles.data(), sizeof(*triangles_gpu) * triangles.size(), cudaMemcpyHostToDevice));

    return nodes_gpu;
}
