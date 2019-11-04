cmake_minimum_required(VERSION 3.12)
project(raytracer LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|GNU")
    message("Clang or G++ detected, enabling optimizations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -fno-trapping-math -fno-signed-zeros -flto")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_definitions(-D_USE_MATH_DEFINES)

add_executable(raytracer
    src/kdtree.cc src/triangle.cu src/material.cc src/parse.cc src/light.cc
    src/matrix.cc
    src/vector.cu src/main.cu  src/upload.cu src/kdtree.cu
)
set_target_properties(raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(raytracer PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
target_link_libraries(raytracer SDL2 SDL2_image)
