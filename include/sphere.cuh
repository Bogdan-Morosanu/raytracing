#pragma once

#include "ray.cuh"

namespace rt {
  class Sphere {
  public:
    __host__ __device__ Sphere() = default;
    
    __host__ __device__ Sphere(const Eigen::Vector3f &c, float r)
      : center_(c)
      , radius_(r)
    { }

    __device__ bool is_hit(const Ray &ray) const
    {
      Eigen::Vector3f oc = ray.origin() - center_;
      float a = ray.direction().dot(ray.direction());
      float b = 2.0f * oc.dot(ray.direction());
      float c = oc.dot(oc) - radius_ * radius_;
      float discriminant = b * b - 4 * a * c;
      return discriminant > 0.0f;
    }
    
  private:
    Eigen::Vector3f center_;
    float radius_;
  };
}
