# Cuda Raytracer

This is a basic RayTracer on GPU. We used Cuda to build a KD-Tree on GPU, and to do the major part of rendering on GPU.

Instructions (on Linux)
Requires CMake 3.13+

## Instructions

### Compilaton:
./configure
cd cmake-build && cmake --build . --config Release -j; cd ..

Please note that you can specify your arch:, e.g. for NVIDIA GeForce RTX 2080:
./configure sm_75

### Usage
./cmake-build/raytracer <scene> [outfile]

### Example
./cmake-build/raytracer examples/scenes/iron.json
feh out.ppm

## Authors

Clement Fang, Thibault Guisnet and Anis Ladram, EPITA students (IMAGE 2020)