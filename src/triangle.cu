#include "triangle.hh"

bool Triangle::intersect(Ray &ray,
                         float &dist) const
{
    const Vector edge1 = vertices[1] - vertices[0];
    const Vector edge2 = vertices[2] - vertices[0];
    const Vector h = ray.dir.cross_product(edge2);

    float det = edge1.dot_product(h);
    if (det > -EPSILON && det < EPSILON)
        return false;    // This ray is parallel to this triangle.
    float f = 1.f / det;
    Vector s = ray.o - vertices[0];
    float u = f * (s.dot_product(h));

    s.cross_product_inplace(edge1);
    float v = f * (ray.dir.dot_product(s));
    
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * edge2.dot_product(s);

    if (u < 0.f || u > 1.f || v < 0.f || u + v > 1.f || t <= EPSILON 
                || (dist >= 0.f && t >= dist))
        return false;

    ray.u = u;
    ray.v = v;
    dist = t;
    return true;
}
