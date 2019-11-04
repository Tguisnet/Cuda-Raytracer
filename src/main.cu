#include <fstream>
#include <vector>
#include <omp.h>
#include <chrono>

#include "parse.hh"
#include "vector.hh"
#include "upload.hh"

using namespace std::chrono;

__global__ void render(Pixel *d_vect, KdNodeGpu *d_tree, Material *d_materials,
                       Vector *a_light, Light *d_lights, Vector *d_u, Vector *d_v,
                       Vector *d_center, Vector *d_cam_pos, 
                       unsigned width, unsigned height)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= width || j >= height)
        return;

    Vector o = *d_u * (j - static_cast<int>(height) / 2);
    Vector b = *d_v * (i - static_cast<int>(width) / 2);

    o += *d_center;
    o += b;

    Vector dir = (o - *d_cam_pos).norm_inplace();
    Ray ray(*d_cam_pos, dir);

    d_vect[i * width + j] = direct_light(d_tree, ray, d_materials, a_light, d_lights);
}

int main(int argc, char *argv[])
{
    std::string path_scene;
    std::string out_file = "out";

    if (argc > 1)
        path_scene = argv[1];
    else
    {
        std::cerr << "Usage: " << argv[0] << " <scene> <outfile>\n";
        return 1;
    }

    if (argc > 2)
        out_file = argv[2];

    auto t1 = high_resolution_clock::now();
    Scene scene = parse_scene(path_scene);

    Vector u_n = scene.cam_u.norm_inplace();
    Vector v_n = scene.cam_v.norm_inplace();
    Vector w = v_n.cross_product(u_n);

    float val = tan(scene.fov * M_PI / 360);
    float l = scene.width / 2;
    l /= val;
    Vector center = scene.cam_pos + (w * l); // center

    std::vector<Triangle> vertices;
    for (const auto& name : scene.objs)
      obj_to_vertices(name, scene.mat_names, vertices, scene);
    auto t2 = high_resolution_clock::now();
    std::cout << "Time to parse file: " << duration_cast<duration<double>>(t2 - t1).count() 
              << "s" << std::endl;

    t1 = high_resolution_clock::now();
    auto tree = KdTree(vertices.begin(), vertices.end());
    std::cout << tree.size() << std::endl;

    t2 = high_resolution_clock::now();
    std::cout << "Time to build tree: " << duration_cast<duration<double>>(t2 - t1).count()  << "s" << std::endl;

    KdNodeGpu *d_tree = upload_kd_tree(tree, vertices);
    t1 = high_resolution_clock::now();
    std::cout << "Time to upload in gpu: " << duration_cast<duration<double>>(t1 - t2).count()  << "s" << std::endl;
    
    Pixel *d_vect;
    Vector *d_u;
    Vector *d_v;
    Vector *d_center;
    Vector *d_cam_pos;
    Material *d_materials;
    Vector *a_light;
    Light *d_lights;

   cudaCheckError(cudaMalloc(&d_materials, scene.mat_names.size() * sizeof(Material)));
   cudaCheckError(cudaMemcpy(d_materials, scene.materials, 
               scene.mat_names.size() * sizeof(Material), cudaMemcpyHostToDevice));

   cudaCheckError(cudaMalloc(&d_vect, scene.width * scene.height * sizeof(*d_vect)));
   cudaCheckError(cudaMalloc(&d_u, sizeof(struct Vector)));
   cudaCheckError(cudaMalloc(&d_v, sizeof(struct Vector)));
   cudaCheckError(cudaMalloc(&d_center, sizeof(struct Vector)));
   cudaCheckError(cudaMalloc(&d_cam_pos, sizeof(struct Vector)));

   cudaCheckError(cudaMalloc(&a_light, sizeof(Vector)));
   cudaCheckError(cudaMemcpy(a_light, &scene.a_light, sizeof(Vector), cudaMemcpyHostToDevice));

   auto nb_lights = std::min(scene.lights.size(), MAX_LIGHTS);

   cudaCheckError(cudaMalloc(&d_lights, sizeof(Light) * MAX_LIGHTS));
   cudaCheckError(cudaMemcpy(d_lights, scene.lights.data(), 
                  sizeof(Light) * nb_lights, cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemset(d_lights + nb_lights, 0, (MAX_LIGHTS - nb_lights) * sizeof(Light)));

   cudaCheckError(cudaMemcpy(d_u, &u_n, sizeof(struct Vector), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_v, &v_n, sizeof(struct Vector), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_center, &center, sizeof(struct Vector), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_cam_pos, &scene.cam_pos, 
                             sizeof(struct Vector), cudaMemcpyHostToDevice));

    constexpr int tx = 8;
    constexpr int ty = 64;

    dim3 dim_block(scene.width / tx + (scene.width % tx != 0),
                   scene.height / ty + (scene.height % ty != 0));
    dim3 dim_thread(tx, ty);

    t1 = high_resolution_clock::now();
    render<<<dim_block, dim_thread >>>(d_vect, d_tree, d_materials, a_light, d_lights, d_u, d_v, d_center, d_cam_pos,
                                      scene.width, scene.height);

    std::vector<Pixel> vect(scene.width * scene.height);
    cudaCheckError(cudaMemcpy(vect.data(), d_vect, vect.size() * sizeof(*d_vect),
                   cudaMemcpyDeviceToHost));

    t2 = high_resolution_clock::now();
    std::cout << "Time to ray tracer: " << duration_cast<duration<double>>(t2 - t1).count() << "s\n";

    cudaFree(d_materials);
    cudaFree(a_light);
    cudaFree(d_lights);
    cudaFree(d_vect);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_center);
    cudaFree(d_cam_pos);

    write_ppm(out_file + ".ppm", vect.data(), scene.width, scene.height);
}
