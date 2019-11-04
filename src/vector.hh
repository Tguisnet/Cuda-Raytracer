#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cmath>
#include <iostream>

inline float to_rad(int deg)
{
    return deg * (M_PI / 180.0);
}

class Vector
{
public:
    CUDA_HOSTDEV Vector() {}

    CUDA_HOSTDEV Vector(float x, float y, float z)
    {
      tab[0] = x;
      tab[1] = y;
      tab[2] = z;
    }

    bool is_not_null()
    {
        return tab[0] == 0 || tab[1] == 0 || tab[2] == 0;
    }
    
    CUDA_HOSTDEV Vector operator+(const Vector &rhs) const;
    CUDA_HOSTDEV Vector operator+=(const Vector &rhs);
    CUDA_HOSTDEV Vector operator-(const Vector &rhs) const;
    CUDA_HOSTDEV Vector operator-=(const Vector &rhs);
    CUDA_HOSTDEV Vector operator*(float lambda) const;
    CUDA_HOSTDEV Vector operator*=(float lambda);
    CUDA_HOSTDEV Vector operator*(const Vector &rhs) const;
    CUDA_HOSTDEV Vector operator*=(const Vector &rhs);
    CUDA_HOSTDEV Vector operator/(const Vector &rhs) const;
    CUDA_HOSTDEV Vector operator/=(const Vector &rhs);
    CUDA_HOSTDEV Vector operator/(float lambda) const;
    CUDA_HOSTDEV Vector operator/=(float lambda);

    CUDA_HOSTDEV  float operator[](unsigned idx) const { return tab[idx]; };
    CUDA_HOSTDEV  float& operator[](unsigned idx) { return tab[idx]; };

   // CUDA_HOSTDEV Vector operator*(float lambda, const Vector &rhs);

    CUDA_HOSTDEV Vector cross_product(const Vector &rhs) const;
    CUDA_HOSTDEV Vector cross_product_inplace(const Vector &rhs);
    CUDA_HOSTDEV Vector norm(void) const;
    CUDA_HOSTDEV Vector norm_inplace(void);

     CUDA_HOSTDEV float dot_product(const Vector &rhs) const;

     CUDA_HOSTDEV float get_dist() { return sqrtf(tab[0] * tab[0] + tab[1] * tab[1] + tab[2] * tab[2]); };

     CUDA_HOSTDEV void set(float x, float y, float z)
    {
        tab[0] = x;
        tab[1] = y;
        tab[2] = z;
    }

    friend std::ostream& operator <<(std::ostream& os, const Vector &v);

    float tab[3];
};

CUDA_HOSTDEV Vector operator/(float lambda, const Vector &v);
CUDA_HOSTDEV Vector operator*(float lambda, const Vector &v);

struct Pixel
{
    Pixel() = default;
    CUDA_HOSTDEV Pixel(uint8_t x, uint8_t y, uint8_t z) : x(x), y(y), z(z) {}
    CUDA_HOSTDEV Pixel(const Vector &vect)
    {
        x = fminf(vect[0], 1.f) * 255;
        y = fminf(vect[1], 1.f) * 255;
        z = fminf(vect[2], 1.f) * 255;
    }

    uint8_t x;
    uint8_t y;
    uint8_t z;
};
